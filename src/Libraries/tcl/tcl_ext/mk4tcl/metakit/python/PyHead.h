/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 21, 2023.
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

// PyHead.h --
// $Id: PyHead.h 1230 2007-03-09 15:58:53Z jcw $
// This is part of MetaKit, see http://www.equi4.com/metakit.html
// Copyright (C) 1999-2004 Gordon McMillan and Jean-Claude Wippler.
//
//  Common object header class

#if !defined INCLUDE_PYHEAD_H
#define INCLUDE_PYHEAD_H

#include <Python.h>

class PyHead: public PyObject {
  public:
    PyHead(PyTypeObject &t) {
#ifdef Py_TRACE_REFS
        _ob_next = 0;
        _ob_prev = 0;
        _Py_AddToAllObjects(this, 0);
#endif 
        ob_refcnt = 1;
        ob_type = &t;
    }
};

#endif
