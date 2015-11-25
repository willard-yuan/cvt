#include <boost/python.hpp>
#include <iostream>
#include <vector>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>

#include "hahog.cc"

#if (PY_VERSION_HEX < 0x03000000)
static void numpy_import_array_wrapper()
#else
static int* numpy_import_array_wrapper()
#endif
{
  /* Initialise numpy API and use 2/3 compatible return */
  import_array();
}

BOOST_PYTHON_MODULE(csfm) {
  using namespace boost::python;

  //google::InitGoogleLogging("csfm");
  boost::python::numeric::array::set_module_and_type("numpy", "ndarray");
  numpy_import_array_wrapper();


  def("hahog", csfm::hahog,
      (boost::python::arg("peak_threshold") = 0.003,
       boost::python::arg("edge_threshold") = 10,
       boost::python::arg("target_num_features") = 0,
       boost::python::arg("use_adaptive_suppression") = false
      )
  );
}
