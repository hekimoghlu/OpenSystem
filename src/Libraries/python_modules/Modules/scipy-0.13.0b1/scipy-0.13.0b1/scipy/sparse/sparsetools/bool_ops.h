/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 22, 2023.
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

#ifndef BOOL_OPS_H
#define BOOL_OPS_H
/*
 * Functions to handle arithmetic operations on NumPy Bool values.
 */
#include <numpy/arrayobject.h>
#include <assert.h>

/*
 * A compiler time (ct) assert macro from 
 * http://www.pixelbeat.org/programming/gcc/static_assert.html
 * This is used to assure that npy_bool_wrapper is the right size.
 */
#define ct_assert(e) extern char (*ct_assert(void)) [sizeof(char[1 - 2*!(e)])]

class npy_bool_wrapper {
    public:
        char value;
        
        /* operators */
        operator char() const {
            if(value != 0) {
                return 1;
            } else {
                return 0;
            }
        }
        npy_bool_wrapper& operator=(const npy_bool_wrapper& x) {
            value = x;
            return (*this);
        }
        npy_bool_wrapper operator+(const npy_bool_wrapper& x) {
            return (x || value) ? 1 : 0;
        }
        /* inplace operators */
        npy_bool_wrapper operator+=(const npy_bool_wrapper& x) {
            value = (x || value) ? 1 : 0;
            return (*this);
        }
        npy_bool_wrapper operator*=(const npy_bool_wrapper& x) {
            value = (value && x) ? 1 : 0;
            return (*this);
        }
        /* constructors */
        npy_bool_wrapper() { 
            value = 0; 
        }
        template <class T>
        npy_bool_wrapper(T x) {
            value = (x) ? 1 : 0;
        }
};

ct_assert(sizeof(char) == sizeof(npy_bool_wrapper));

#endif
