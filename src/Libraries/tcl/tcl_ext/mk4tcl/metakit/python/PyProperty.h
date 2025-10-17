/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 10, 2023.
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

// PyProperty.h --
// $Id: PyProperty.h 1230 2007-03-09 15:58:53Z jcw $
// This is part of MetaKit, see http://www.equi4.com/metakit.html
// Copyright (C) 1999-2004 Gordon McMillan and Jean-Claude Wippler.
//
//  Property class header

#if !defined INCLUDE_PYPROPERTY_H
#define INCLUDE_PYPROPERTY_H

#include <mk4.h>
#include "PyHead.h"

#define PyProperty_Check(ob) ((ob)->ob_type == &PyPropertytype)

extern PyTypeObject PyPropertytype;

class PyProperty: public PyHead, public c4_Property {
  public:
    //PyProperty();
    PyProperty(const c4_Property &o): PyHead(PyPropertytype), c4_Property(o){}
    PyProperty(char t, const char *n): PyHead(PyPropertytype), c4_Property(t, n)
      {}
    ~PyProperty(){}
};

PyObject *PyProperty_new(PyObject *o, PyObject *_args);

#endif
