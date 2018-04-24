#include "SaveBinaryCV.h"

namespace cv {

	//! Save cv::Mat as binary
	bool SaveMatBinary(const std::string& filename, const cv::Mat& output) {
		std::ofstream ofs(filename, std::ios::binary);
		if (!ofs.is_open())
			return false;
		writeMatBinary(ofs, output);
		return true;
	}

	//! Write cv::Mat as binary
	void writeMatBinary(std::ofstream& ofs, const cv::Mat& out_mat)
	{
		if (out_mat.empty()) {
			int s = 0;
			ofs.write((const char*)(&s), sizeof(int));
			return;
		}
		int type = out_mat.type();
		ofs.write((const char*)(&out_mat.rows), sizeof(int));
		ofs.write((const char*)(&out_mat.cols), sizeof(int));
		ofs.write((const char*)(&type), sizeof(int));
		ofs.write((const char*)(out_mat.data), out_mat.elemSize() * out_mat.total());
	}


	//! Load cv::Mat as binary
	bool LoadMatBinary(const std::string& filename, cv::Mat& output) {
		std::ifstream ifs(filename, std::ios::binary);
		if (!ifs.is_open())
			return false;
		readMatBinary(ifs, output);
		return true;
	}

	//! Read cv::Mat from binary
	void readMatBinary(std::ifstream& ifs, cv::Mat& in_mat)
	{
		int rows, cols, type;
		ifs.read((char*)(&rows), sizeof(int));
		if (rows == 0) {
			return;
		}
		ifs.read((char*)(&cols), sizeof(int));
		ifs.read((char*)(&type), sizeof(int));

		in_mat.release();
		in_mat.create(rows, cols, type);
		ifs.read((char*)(in_mat.data), in_mat.elemSize() * in_mat.total());
	}


	void writeKeyPointBinary(std::ofstream& ofs, const cv::KeyPoint& key_point) {
		ofs.write((const char*)(&key_point.angle), sizeof(float));
		ofs.write((const char*)(&key_point.size), sizeof(float));
		ofs.write((const char*)(&key_point.response), sizeof(float));
		ofs.write((const char*)(&key_point.class_id), sizeof(int));
		ofs.write((const char*)(&key_point.octave), sizeof(int));
		writePointBinary<float>(ofs, key_point.pt);
	}


	void readKeyPointBinary(std::ifstream& ifs, cv::KeyPoint& key_point)
	{
		ifs.read((char*)(&key_point.angle), sizeof(float));
		ifs.read((char*)(&key_point.size), sizeof(float));
		ifs.read((char*)(&key_point.response), sizeof(float));
		ifs.read((char*)(&key_point.class_id), sizeof(int));
		ifs.read((char*)(&key_point.octave), sizeof(int));
		readPointBinary<float>(ifs, key_point.pt);
	}


	void writeKeyPointsBinary(std::ofstream& ofs, const std::vector<cv::KeyPoint>& key_point) {
		size_t num_pts = key_point.size();
		ofs.write((const char*)&num_pts, sizeof(size_t));
		for (size_t i = 0; i < num_pts;i++) {
			writeKeyPointBinary(ofs, key_point[i]);
		}
	}


	void readKeyPointsBinary(std::ifstream& ifs, std::vector<cv::KeyPoint>& key_point) {
		size_t num_pts;
		ifs.read((char*)&num_pts, sizeof(size_t));
		key_point.resize(num_pts);
		for (size_t i = 0; i < num_pts;i++) {
			readKeyPointBinary(ifs, key_point[i]);
		}
	}


}