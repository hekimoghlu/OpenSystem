/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 8, 2022.
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

// PyStorage.h --
// $Id: PyStorage.h 1230 2007-03-09 15:58:53Z jcw $
// This is part of MetaKit, see http://www.equi4.com/metakit.html
// Copyright (C) 1999-2004 Gordon McMillan and Jean-Claude Wippler.
//
//  Storage class header

#if !defined INCLUDE_PYSTORAGE_H
#define INCLUDE_PYSTORAGE_H

#include <mk4.h>
#include "PyHead.h"

extern PyTypeObject PyStoragetype;
class SiasStrategy;

#define PyStorage_Check(v) ((v)->ob_type==&PyStoragetype)

class PyStorage: public PyHead, public c4_Storage {
  public:
    PyStorage(): PyHead(PyStoragetype){}
    PyStorage(c4_Strategy &strategy_, bool owned_ = false, int mode_ = 1):
      PyHead(PyStoragetype), c4_Storage(strategy_, owned_, mode_){}
    PyStorage(const char *fnm, int mode): PyHead(PyStoragetype), c4_Storage(fnm,
      mode){}
    PyStorage(const c4_Storage &storage_): PyHead(PyStoragetype), c4_Storage
      (storage_){}
    //  PyStorage(const char *fnm, const char *descr) 
    //    : PyHead(PyStoragetype), c4_Storage(fnm, descr) { }
    ~PyStorage(){}
};

#endif
