// Copyright (c) 2022, ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: Johannes L. Schoenberger (jsch-at-demuc-dot-de)

#include "exe/vocab_tree.h"

#include <numeric>

#include "base/database.h"
#include "feature/matching.h"
#include "feature/sift.h"
#include "feature/utils.h"
#include "retrieval/visual_index.h"
#include "util/misc.h"
//#include "util/opengl_utils.h"

namespace colmap {
namespace {

std::vector<Image> ReadVocabTreeRetrievalImageList(const std::string& path,
                                                   Database* database) {
  std::vector<Image> images;
  if (path.empty()) {
    images.reserve(database->NumImages());
    for (const auto& image : database->ReadAllImages()) {
      images.push_back(image);
    }
  } else {
    DatabaseTransaction database_transaction(database);

    const auto image_names = ReadTextFileLines(path);
    images.reserve(image_names.size());
    for (const auto& image_name : image_names) {
      const auto image = database->ReadImageWithName(image_name);
      std::cout << "image.ImageId(): " << image.ImageId() << std::endl;
      std::cout << "kInvalidImageId: " << kInvalidImageId << std::endl;
      CHECK_NE(image.ImageId(), kInvalidImageId);
      images.push_back(image);
    }
  }
  return images;
}

}  // namespace

int RunVocabTreeRetriever(int argc, char** argv) {
  std::string vocab_tree_path = "/Users/willard/colmap_index/vocab_tree_flickr100K_words1M.bin";
  std::string database_image_list_path = "";
  std::string database_path = "/Users/willard/colmap_index/database.db";
  std::string query_image_list_path = "/Users/willard/colmap_index/query_image_list";
  std::string output_index_path = "/Users/willard/colmap_index/index.bin";
  retrieval::VisualIndex<>::QueryOptions query_options;
  query_options.num_images_after_verification = 10;
  int max_num_features = -1;

  std::cout << "vocab_tree_path: " << vocab_tree_path << std::endl;
  std::cout << "database_image_list_path: " << database_image_list_path << std::endl;
  std::cout << "query_image_list_path: " << query_image_list_path << std::endl;
  std::cout << "output_index_path: " << output_index_path << std::endl; 

  retrieval::VisualIndex<> visual_index;
  visual_index.Read(vocab_tree_path);

  Database database(database_path);

  const auto database_images =
      ReadVocabTreeRetrievalImageList(database_image_list_path, &database);
  const auto query_images =
      (!query_image_list_path.empty() || output_index_path.empty())
          ? ReadVocabTreeRetrievalImageList(query_image_list_path, &database)
          : std::vector<Image>();

  std::cout << "database_images size: " << database_images.size() << std::endl;
  std::cout << "query_images: " << query_images.size() << std::endl;

  //////////////////////////////////////////////////////////////////////////////
  // Perform image indexing
  //////////////////////////////////////////////////////////////////////////////

  for (size_t i = 0; i < database_images.size(); ++i) {
    Timer timer;
    timer.Start();

    std::cout << StringPrintf("Indexing image [%d/%d]", i + 1,
                              database_images.size())
              << std::flush;

    if (visual_index.ImageIndexed(database_images[i].ImageId())) {
      std::cout << std::endl;
      continue;
    }

    auto keypoints = database.ReadKeypoints(database_images[i].ImageId());
    auto descriptors = database.ReadDescriptors(database_images[i].ImageId());
    if (max_num_features > 0 && descriptors.rows() > max_num_features) {
      ExtractTopScaleFeatures(&keypoints, &descriptors, max_num_features);
    }

    visual_index.Add(retrieval::VisualIndex<>::IndexOptions(),
                     database_images[i].ImageId(), keypoints, descriptors);

    std::cout << StringPrintf(" in %.3fs", timer.ElapsedSeconds()) << std::endl;
  }

  // Compute the TF-IDF weights, etc.
  visual_index.Prepare();

  // Optionally save the indexing data for the database images (as well as the
  // original vocabulary tree data) to speed up future indexing.
  if (!output_index_path.empty()) {
    visual_index.Write(output_index_path);
  }

  if (query_images.empty()) {
    return EXIT_SUCCESS;
  }

  //////////////////////////////////////////////////////////////////////////////
  // Perform image queries
  //////////////////////////////////////////////////////////////////////////////

  std::unordered_map<image_t, const Image*> image_id_to_image;
  image_id_to_image.reserve(database_images.size());
  for (const auto& image : database_images) {
    image_id_to_image.emplace(image.ImageId(), &image);
  }

  for (size_t i = 0; i < query_images.size(); ++i) {
    Timer timer;
    timer.Start();

    std::cout << StringPrintf("Querying for image %s [%d/%d]",
                              query_images[i].Name().c_str(), i + 1,
                              query_images.size())
              << std::flush;

    auto keypoints = database.ReadKeypoints(query_images[i].ImageId());
    auto descriptors = database.ReadDescriptors(query_images[i].ImageId());
    if (max_num_features > 0 && descriptors.rows() > max_num_features) {
      ExtractTopScaleFeatures(&keypoints, &descriptors, max_num_features);
    }

    std::vector<retrieval::ImageScore> image_scores;
    visual_index.Query(query_options, keypoints, descriptors, &image_scores);

    std::cout << StringPrintf(" in %.3fs", timer.ElapsedSeconds()) << std::endl;
    for (const auto& image_score : image_scores) {
      const auto& image = *image_id_to_image.at(image_score.image_id);
      std::cout << StringPrintf("  image_id=%d, image_name=%s, score=%f",
                                image_score.image_id, image.Name().c_str(),
                                image_score.score)
                << std::endl;
    }
  }

  return EXIT_SUCCESS;
}

}  // namespace colmap
