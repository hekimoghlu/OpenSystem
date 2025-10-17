/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 5, 2025.
 *
 * Licensed under the Apache License, Version 2.0 (the ""License"");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an ""AS IS"" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please contact NeXTHub Corporation, 651 N Broad St, Suite 201, 
 * Middletown, DE 19709, New Castle County, USA.
 *
 */

//
//  IOHIDAccelerationAlgorithm.hpp
//  IOHIDFamily
//
//  Created by YG on 10/29/15.
//
//

#ifndef IOHIDAccelerationAlgorithm_hpp
#define IOHIDAccelerationAlgorithm_hpp

#include <CoreFoundation/CoreFoundation.h>
#include "IOHIDAccelerationTable.hpp"
#include <vector>
#include <set>
#include <ostream>
#include <iomanip>
#include "CF.h"

#define FIXED_TO_DOUBLE(x) ((x)/65536.0)
#define DOUBLE_TO_FIXED(x) (uint64_t)((x)*65536.0)
#define MAX_DEVICE_THRESHOLD   FIXED_TO_DOUBLE(0x7fffffff)
#define kCursorScale (96.0/67.0)

class IOHIDAccelerationAlgorithm {
   
public:

    enum Type {
      Table,
      Parametric,
      Unknown
    };
  
    IOHIDAccelerationAlgorithm () {}
  
    virtual ~IOHIDAccelerationAlgorithm() {}
    virtual double multiplier (double value) {
      return value;
    }
    virtual IOHIDAccelerationAlgorithm::Type getType () const {
      return IOHIDAccelerationAlgorithm::Unknown;
    };
  
    virtual void serialize (CFMutableDictionaryRef dict) const = 0;
  
};



class IOHIDParametricAcceleration : public  IOHIDAccelerationAlgorithm {

public:

    typedef struct ACCELL_CURVE {
        double  Index;
        double  GainLinear;
        double  GainParabolic;
        double  GainCubic;
        double  GainQudratic;
        double  TangentSpeedLinear;
        double  TangentSpeedParabolicRoot;
        bool isValid () {
          return GainLinear || GainParabolic  || GainCubic || GainQudratic ;
        }
    } ACCELL_CURVE;

    double        tangent [2];
    double        m [2];
    double        b [2];
    
    ACCELL_CURVE  accel;
    
    double        resolution_;
    double        rate_;
    double        accelIndex_;

    static double GetCurveParameter (CFDictionaryRef curve, CFStringRef parameter);
    static ACCELL_CURVE GetCurve (CFDictionaryRef curve);
  
public:
  
    static  IOHIDParametricAcceleration * CreateWithParameters (CFArrayRef curves, double acceleration, double resolution, double rate);
    
    virtual ~IOHIDParametricAcceleration() {}
  
    virtual double multiplier (double value);

    virtual IOHIDAccelerationAlgorithm::Type getType () const {
        return IOHIDAccelerationAlgorithm::Parametric;
    };

    virtual void serialize (CFMutableDictionaryRef dict) const;

protected:

    IOHIDParametricAcceleration () {};
};


class IOHIDTableAcceleration : public  IOHIDAccelerationAlgorithm {
  
protected:
  
    std::vector <ACCEL_SEGMENT> segments_;
    
    double resolution_;
    double rate_;
    
    ACCEL_POINT  InterpolatePoint (const ACCEL_POINT &p, const ACCEL_POINT &p0, const ACCEL_POINT &p1, double scale);
    static ACCEL_POINT  InterpolatePoint (const ACCEL_POINT &p, const ACCEL_POINT &p0, const ACCEL_POINT &p1,  double scale , boolean_t isLower);
   
    void  InterpolateFunction (const ACCEL_TABLE_ENTRY *lo, const ACCEL_TABLE_ENTRY *hi, double scale, std::set<ACCEL_POINT> &result);
  
public:
  
    static  IOHIDTableAcceleration * CreateWithTable (CFDataRef table, double acceleration, double resolution, double rate);
    static  IOHIDTableAcceleration * CreateOriginalWithTable (CFDataRef table, double acceleration, double resolution, double rate);
    
    virtual ~IOHIDTableAcceleration() {}
    
    virtual double multiplier (double value);

    virtual IOHIDAccelerationAlgorithm::Type getType () const {
        return IOHIDAccelerationAlgorithm::Table;
    };

    virtual void serialize (CFMutableDictionaryRef dict) const;

protected:
  
  IOHIDTableAcceleration () {};
};

#endif /* IOHIDAccelerationAlgorithm_hpp */
