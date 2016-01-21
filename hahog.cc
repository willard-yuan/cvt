#include "types.h"
#include "hahog.h"

#include <vector>
#include <iostream>

extern "C" {
#include "vl/covdet.h"
#include "vl/sift.h"
#include "vl/generic.h"
#include "vl/host.h"
#include <time.h>
}

using namespace std;


namespace csfm {

bp::object hahog(PyObject *image, int target_num_features, bool use_adaptive_suppression, bool verbose) {

    PyArrayContiguousView<float> im((PyArrayObject *)image);

    typedef enum _VlCovDetDescriptorType{
        VL_COVDET_DESC_SIFT
    } VlCovDetDescriporType;


    VlCovDetMethod method = VL_COVDET_METHOD_DOG;
    vl_bool estimateAffineShape = VL_FALSE;
    vl_bool estimateOrientation = VL_FALSE;

    vl_bool doubleImage = VL_TRUE;
    vl_index octaveResolution = -1;
    double edgeThreshold = 10;
    double peakThreshold = 0.01;
    double lapPeakThreshold = -1;

    int descriptorType = -1;
    vl_index patchResolution = -1;
    double patchRelativeExtent = -1;
    double patchRelativeSmoothing = -1;

    double boundaryMargin = 2.0;

    if (descriptorType < 0) descriptorType = VL_COVDET_DESC_SIFT;

    switch (descriptorType){
        case VL_COVDET_DESC_SIFT:
            if (patchResolution < 0) patchResolution = 15;
            if (patchRelativeExtent < 0) patchRelativeExtent = 7.5;
            if (patchRelativeSmoothing <0) patchRelativeSmoothing = 1;
            cout << "vl_covdet: patchRelativeExtent " << patchRelativeExtent << endl;
    }


    if (im.valid()) {
        clock_t t_start = clock();
        // create a detector object: VL_COVDET_METHOD_HESSIAN
        VlCovDet * covdet = vl_covdet_new(method);

        // set various parameters (optional)
        vl_covdet_set_first_octave(covdet, doubleImage? -1 : 0);

        //vl_covdet_set_octave_resolution(covdet, octaveResolution);
        if (octaveResolution >= 0) vl_covdet_set_octave_resolution(covdet, octaveResolution);
        if (peakThreshold >= 0) vl_covdet_set_peak_threshold(covdet, peakThreshold);
        if (edgeThreshold >= 0) vl_covdet_set_edge_threshold(covdet, edgeThreshold);
        if (lapPeakThreshold >= 0) vl_covdet_set_laplacian_peak_threshold(covdet, lapPeakThreshold);

        //vl_covdet_set_target_num_features(covdet, target_num_features);
        //vl_covdet_set_use_adaptive_suppression(covdet, use_adaptive_suppression);

        if(verbose){
            std::cout << "vl_covdet: doubling image: " << VL_YESNO(vl_covdet_get_first_octave(covdet) < 0) << endl;
        }

        if (verbose) {
            cout << "vl_covdet: detector: " << vl_enumeration_get_by_value(vlCovdetMethods, method)->name << endl;
            cout << "vl_covdet: peak threshold: " << vl_covdet_get_peak_threshold(covdet) << ", edge threshold: " << vl_covdet_get_edge_threshold(covdet) << endl;
        }
        // process the image and run the detector
        vl_covdet_put_image(covdet, im.data(), im.shape(1), im.shape(0));
        clock_t t_scalespace = clock();
        vl_covdet_detect(covdet);
        clock_t t_detect = clock();

        if (verbose) {
            vl_size numFeatures = vl_covdet_get_num_features(covdet) ;
            cout << "vl_covdet: " << vl_covdet_get_num_non_extrema_suppressed(covdet) << " features suppressed as duplicate (threshold: "
            << vl_covdet_get_non_extrema_suppression_threshold(covdet) << ")"<< endl;
            cout << "vl_covdet: detected " << numFeatures << " features" << endl;
        }


        //drop feature on the margin(optimal)
        if(boundaryMargin > 0){
            vl_covdet_drop_features_outside(covdet, boundaryMargin);
            if(verbose){
                vl_size numFeatures = vl_covdet_get_num_features(covdet);
                cout << "vl_covdet: kept " << numFeatures << " inside the boundary margin "<< boundaryMargin << endl;
            }
        }

        /* affine adaptation if needed */
        bool estimateAffineShape = true;
        if (estimateAffineShape) {
            if (verbose) {
                vl_size numFeaturesBefore = vl_covdet_get_num_features(covdet) ;
                cout << "vl_covdet: estimating affine shape for " << numFeaturesBefore << " features" << endl;
            }

            vl_covdet_extract_affine_shape(covdet) ;

            if (verbose) {
                vl_size numFeaturesAfter = vl_covdet_get_num_features(covdet) ;
                cout << "vl_covdet: "<< numFeaturesAfter << " features passed affine adaptation" << endl;
            }
        }

        // compute the orientation of the features (optional)
        //clock_t t_affine = clock();
        //vl_covdet_extract_orientations(covdet);
        //clock_t t_orient = clock();

        // get feature descriptors
        vl_size numFeatures = vl_covdet_get_num_features(covdet);
        VlCovDetFeature const *feature = (VlCovDetFeature const *)vl_covdet_get_features(covdet);
        VlSiftFilt *sift = vl_sift_new(16, 16, 1, 3, 0);
        vl_index i;
        vl_size dimension = 128;
        vl_size patchSide = 2 * patchResolution + 1;

        std::vector<float> points(6 * numFeatures);
        std::vector<float> desc(dimension * numFeatures);
        std::vector<float> patch(patchSide * patchSide);
        std::vector<float> patchXY(2 * patchSide * patchSide);

        double patchStep = (double)patchRelativeExtent / patchResolution;

        if (verbose) {
            cout << "vl_covdet: descriptors: type = sift" << ", resolution = " << patchResolution << ", extent = " << patchRelativeExtent << ", smoothing = " << patchRelativeSmoothing << endl;
          }

        vl_sift_set_magnif(sift, 3.0);
        for (i = 0; i < (signed)numFeatures; ++i) {
            points[6 * i + 0] = feature[i].frame.x;
            points[6 * i + 1] = feature[i].frame.y;
            points[6 * i + 2] = feature[i].frame.a11;
            points[6 * i + 3] = feature[i].frame.a12;
            points[6 * i + 4] = feature[i].frame.a21;
            points[6 * i + 5] = feature[i].frame.a22;

            vl_covdet_extract_patch_for_frame(covdet,
                                        &patch[0],
                                        patchResolution,
                                        patchRelativeExtent,
                                        patchRelativeSmoothing,
                                        feature[i].frame);

            vl_imgradient_polar_f(&patchXY[0], &patchXY[1],
                            2, 2 * patchSide,
                            &patch[0], patchSide, patchSide, patchSide);

            vl_sift_calc_raw_descriptor(sift,
                                  &patchXY[0],
                                  &desc[dimension * i],
                                  (int)patchSide, (int)patchSide,
                                  (double)(patchSide - 1) / 2, (double)(patchSide - 1) / 2,
                                  (double)patchRelativeExtent / (3.0 * (4 + 1) / 2) / patchStep,
                                  VL_PI / 2);
            /*cout << "test:" << endl;
            for(vl_index j = 0; j < 128; ++j){
                cout << desc[dimension * i+j] << "\t";
            }
            cout << endl;*/
        }
        vl_sift_delete(sift);
        vl_covdet_delete(covdet);

        clock_t t_description = clock();
        // std::cout << "t_scalespace " << float(t_scalespace - t_start)/CLOCKS_PER_SEC << "\n";
        // std::cout << "t_detect " << float(t_detect - t_scalespace)/CLOCKS_PER_SEC << "\n";
        // std::cout << "t_affine " << float(t_affine - t_detect)/CLOCKS_PER_SEC << "\n";
        // std::cout << "t_orient " << float(t_orient - t_affine)/CLOCKS_PER_SEC << "\n";
        // std::cout << "description " << float(t_description - t_orient)/CLOCKS_PER_SEC << "\n";

        bp::list retn;
        npy_intp points_shape[2] = {npy_intp(numFeatures), 6};
        retn.append(bpn_array_from_data(2, points_shape, &points[0]));
        npy_intp desc_shape[2] = {npy_intp(numFeatures), npy_intp(dimension)};
        retn.append(bpn_array_from_data(2, desc_shape, &desc[0]));
        return retn;
    }
    return bp::object();
}

}
