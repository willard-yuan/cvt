#ifndef __SAVE_BINARY_CV__
#define __SAVE_BINARY_CV__

#include <opencv2/core.hpp>
#include <fstream>

namespace cv {

	//! Save cv::Mat as binary
	/*!
	@param[in] filename filaname to save
	@param[in] output cvmat to save
	*/
	bool SaveMatBinary(const std::string& filename, const cv::Mat& output);


	//! Load cv::Mat as binary
	/*!
	@param[in] filename filaname to load
	@param[out] output loaded cv::Mat
	*/
	bool LoadMatBinary(const std::string& filename, cv::Mat& output);

	//! Write cv::Mat as binary
	/*!
	@param[out] ofs output file stream
	@param[in] out_mat mat to save
	*/
	void writeMatBinary(std::ofstream& ofs, const cv::Mat& out_mat);


	//! Read cv::Mat from binary
	/*!
	@param[in] ifs input file stream
	@param[out] in_mat mat to load
	*/
	void readMatBinary(std::ifstream& ifs, cv::Mat& in_mat);

	//! Write cv::KeyPoint as binary
	/*!
	@param[in] ofs output file stream
	@param[in] key_point cv::KeyPoint to save
	*/
	void writeKeyPointBinary(std::ofstream& ofs, const cv::KeyPoint& key_point);

	//! Read cv::KeyPoint from binary
	/*!
	@param[in] ifs input file stream
	@param[out] key_point cv::KeyPoint to load
	*/
	void readKeyPointBinary(std::ifstream& ifs, cv::KeyPoint& key_point);

	//! Write std::vector<cv::KeyPoint> as binary
	/*!
	@param[in] ofs output file stream
	@param[in] key_points std::vector<cv::KeyPoint> to save
	*/
	void writeKeyPointsBinary(std::ofstream& ofs, const std::vector<cv::KeyPoint>& key_point);

	//! Read std::vector<cv::KeyPoint> from binary
	/*!
	@param[in] ifs input file stream
	@param[out] key_points std::vector<cv::KeyPoint> to load
	*/
	void readKeyPointsBinary(std::ifstream& ifs, std::vector<cv::KeyPoint>& key_point);

	//! Write cv::Size_<T> as binary
	/*!
	@param[in] ofs output file stream
	@param[in] size cv::Size_<T> to save
	*/
	template <typename T>
	void writeSizeBinary(std::ofstream& ofs, const cv::Size_<T>& size) {
		ofs.write((const char*)(&size.width), sizeof(T));
		ofs.write((const char*)(&size.height), sizeof(T));
	}

	//! Read cv::Size_<T> from binary
	/*!
	@param[in] ifs input file stream
	@param[out] size cv::Size_<T> to load
	*/
	template <typename T>
	void readSizeBinary(std::ifstream& ifs, cv::Size_<T>& size) {
		ifs.read((char*)(&size.width), sizeof(T));
		ifs.read((char*)(&size.height), sizeof(T));
	}

	//! Write cv::Point_<T> as binary
	/*!
	@param[in] ofs output file stream
	@param[in] pt cv::Point_<T> to save
	*/
	template <typename T>
	void writePointBinary(std::ofstream& ofs, const cv::Point_<T>& pt) {
		ofs.write((const char*)(&pt.x), sizeof(T));
		ofs.write((const char*)(&pt.y), sizeof(T));
	}

	//! Read cv::Point_<T> from binary
	/*!
	@param[in] ifs input file stream
	@param[out] pt cv::Point_<T> to load
	*/
	template <typename T>
	void readPointBinary(std::ifstream& ifs, cv::Point_<T>& pt) {
		ifs.read((char*)(&pt.x), sizeof(T));
		ifs.read((char*)(&pt.y), sizeof(T));
	}

}

#endif