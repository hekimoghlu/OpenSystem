/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 11, 2025.
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
//  IOHIDAccelerationTable.hpp
//  IOHIDFamily
//
//  Created by YG on 10/29/15.
//
//

#ifndef IOHIDAccelerationTable_hpp
#define IOHIDAccelerationTable_hpp

#include <ostream>

#define FIXED_TO_DOUBLE(x) ((x)/65536.0)

typedef struct {
  double   m;
  double   b;
  double   x;
} ACCEL_SEGMENT;

typedef struct {
    double  x;
    double  y;
} ACCEL_POINT;

struct ACCEL_TABLE_ENTRY {

  template<typename T>
  T acceleration () const;

  uint32_t count  () const;
  uint32_t length () const;
  
  template<typename T>
  T x  (unsigned int index) const;
  
  template<typename T>
  T y  (unsigned int index) const;

  ACCEL_POINT point (unsigned int) const;

private:

  uint32_t accel_;
  uint16_t count_;
  uint32_t points_[1][2];
  
} __attribute__ ((packed));


struct ACCEL_TABLE {
 
  template<typename T>
  T scale () const ;
  
  uint32_t count () const;

  uint32_t signature () const;

  const ACCEL_TABLE_ENTRY * entry (int index) const;
  
  friend std::ostream & operator<<(std::ostream &os, const ACCEL_TABLE& t);

private:

  uint32_t scale_;
  uint32_t signature_;
  uint16_t count_;
  ACCEL_TABLE_ENTRY entry_;
  
} __attribute__ ((packed));

#define APPLE_ACCELERATION_MT_TABLE_SIGNATURE       0x2a425355
#define APPLE_ACCELERATION_DEFAULT_TABLE_SIGNATURE  0x30303240

inline  int32_t ACCEL_TABLE_CONSUME_INT32 (const void ** t) {
    int32_t val = OSReadBigInt32(*t, 0);
    *t = (uint8_t*)*t + 4;
    return val;
}

inline  int16_t ACCEL_TABLE_CONSUME_INT16 (const void ** t) {
    int16_t val = OSReadBigInt16(*t, 0);
    *t = (uint8_t*)*t + 2;
    return val;
}

#define ACCELL_TABLE_CONSUME_POINT(t) {FIXED_TO_DOUBLE(ACCEL_TABLE_CONSUME_INT32(t)), FIXED_TO_DOUBLE(ACCEL_TABLE_CONSUME_INT32(t))}

#endif /* IOHIDAccelerationTable_hpp */
