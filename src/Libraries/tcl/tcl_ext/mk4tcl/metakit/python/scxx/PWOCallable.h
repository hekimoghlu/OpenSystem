/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 30, 2024.
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
#if !defined(PWOCALLABLE_H_INCLUDED_)
#define PWOCALLABLE_H_INCLUDED_

#include "PWOBase.h"
#include "PWOSequence.h"
#include "PWOMapping.h"

class PWOCallable: public PWOBase {
  public:
    PWOCallable(): PWOBase(){}
    ;
    PWOCallable(PyObject *obj): PWOBase(obj) {
        _violentTypeCheck();
    };
    virtual ~PWOCallable(){}
    ;
    virtual PWOCallable &operator = (const PWOCallable &other) {
        GrabRef(other);
        return  *this;
    };
    PWOCallable &operator = (const PWOBase &other) {
        GrabRef(other);
        _violentTypeCheck();
        return  *this;
    };
    virtual void _violentTypeCheck() {
        if (!isCallable()) {
            GrabRef(0);
            Fail(PyExc_TypeError, "Not a callable object");
        }
    };
    PWOBase call()const;
    PWOBase call(PWOTuple &args)const;
    PWOBase call(PWOTuple &args, PWOMapping &kws)const;
};

#endif
