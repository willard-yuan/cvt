#ifndef conic_section_h
#define conic_section_h

#include <opencv2/core/core.hpp>

namespace pupiltracker {
    
    template<typename T>
    class ConicSection_
    {
    public:
        T A,B,C,D,E,F;
        
        ConicSection_(cv::RotatedRect r)
        {
            cv::Point_<T> axis((T)std::cos(CV_PI/180.0 * r.angle), (T)std::sin(CV_PI/180.0 * r.angle));
            cv::Point_<T> centre(r.center);
            T a = r.size.width/2;
            T b = r.size.height/2;
            
            initFromEllipse(axis, centre, a, b);
        }
        
        T algebraicDistance(cv::Point_<T> p)
        {
            return A*p.x*p.x + B*p.x*p.y + C*p.y*p.y + D*p.x + E*p.y + F;
        }
        
        T distance(cv::Point_<T> p)
        {
            //    dist
            // -----------
            // |grad|^0.45
            
            T dist = algebraicDistance(p);
            cv::Point_<T> grad = algebraicGradient(p);
            
            T sqgrad = grad.dot(grad);
            
            return dist / std::pow(sqgrad, T(0.45/2));
        }
        
        cv::Point_<T> algebraicGradient(cv::Point_<T> p)
        {
            return cv::Point_<T>(2*A*p.x + B*p.y + D, B*p.x + 2*C*p.y + E);
        }
        
        cv::Point_<T> algebraicGradientDir(cv::Point_<T> p)
        {
            cv::Point_<T> grad = algebraicGradient(p);
            T len = std::sqrt(grad.ddot(grad));
            grad.x /= len;
            grad.y /= len;
            return grad;
        }
        
    protected:
        void initFromEllipse(cv::Point_<T> axis, cv::Point_<T> centre, T a, T b)
        {
            T a2 = a * a;
            T b2 = b * b;
            
            A = axis.x*axis.x / a2 + axis.y*axis.y / b2;
            B = 2*axis.x*axis.y / a2 - 2*axis.x*axis.y / b2;
            C = axis.y*axis.y / a2 + axis.x*axis.x / b2;
            D = (-2*axis.x*axis.y*centre.y - 2*axis.x*axis.x*centre.x) / a2
            + (2*axis.x*axis.y*centre.y - 2*axis.y*axis.y*centre.x) / b2;
            E = (-2*axis.x*axis.y*centre.x - 2*axis.y*axis.y*centre.y) / a2
            + (2*axis.x*axis.y*centre.x - 2*axis.x*axis.x*centre.y) / b2;
            F = (2*axis.x*axis.y*centre.x*centre.y + axis.x*axis.x*centre.x*centre.x + axis.y*axis.y*centre.y*centre.y) / a2
            + (-2*axis.x*axis.y*centre.x*centre.y + axis.y*axis.y*centre.x*centre.x + axis.x*axis.x*centre.y*centre.y) / b2
            - 1;
        }
    };
    typedef ConicSection_<float> ConicSection;
    
} //namespace pupiltracker


#endif /* ConicSection_h */
