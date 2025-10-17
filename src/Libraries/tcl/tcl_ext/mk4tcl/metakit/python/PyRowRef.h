/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 3, 2022.
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

// PyRowRef.h --
// $Id: PyRowRef.h 1230 2007-03-09 15:58:53Z jcw $
// This is part of MetaKit, see http://www.equi4.com/metakit.html
// Copyright (C) 1999-2004 Gordon McMillan and Jean-Claude Wippler.
//
//  RowRef class header

#if !defined INCLUDE_PYROWREF_H
#define INCLUDE_PYROWREF_H

#include <mk4.h>
#include "PyHead.h"
#include <PWOSequence.h>
#include "PyView.h"
#include "PyProperty.h"

#define PyRowRef_Check(ob) ((ob)->ob_type == &PyRowReftype)
#define PyRORowRef_Check(ob) ((ob)->ob_type == &PyRORowReftype)
#define PyGenericRowRef_Check(ob) (PyRowRef_Check(ob) || PyRORowRef_Check(ob))

extern PyTypeObject PyRowReftype;
extern PyTypeObject PyRORowReftype;

class PyRowRef: public PyHead, public c4_RowRef {
  public:
    //PyRowRef();
    PyRowRef(const c4_RowRef &o, int immutable = 0);
    //PyRowRef(c4_Row row);
    ~PyRowRef() {
        c4_Cursor c = &(*(c4_RowRef*)this);
        c._seq->DecRef();
    }
    PyProperty *getProperty(char *nm) {
        c4_View cntr = Container();
        int ndx = cntr.FindPropIndexByName(nm);
        if (ndx >  - 1) {
            return new PyProperty(cntr.NthProperty(ndx));
        }
        return 0;
    };

    PyObject *getPropertyValue(char *nm) {
        PyProperty *prop = getProperty(nm);
        if (prop) {
            PyObject *result = asPython(*prop);
            Py_DECREF(prop);
            return result;
        }
        return 0;
    };

    static void setFromPython(const c4_RowRef &row, const c4_Property &prop,
      PyObject *val);
    static void setDefault(const c4_RowRef &row, const c4_Property &prop);
    PyObject *asPython(const c4_Property &prop);
};

#endif
