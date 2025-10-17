/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 18, 2025.
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

#include "libxml_wrap.h"
#include <libxslt/xslt.h>
#include <libxslt/xsltInternals.h>
#include <libxslt/xsltutils.h>
#include <libxslt/attributes.h>
#include <libxslt/documents.h>
#include <libxslt/extensions.h>
#include <libxslt/extra.h>
#include <libxslt/functions.h>
#include <libxslt/imports.h>
#include <libxslt/keys.h>
#include <libxslt/namespaces.h>
#include <libxslt/numbersInternals.h>
#include <libxslt/pattern.h>
#include <libxslt/preproc.h>
#include <libxslt/templates.h>
#include <libxslt/transform.h>
#include <libxslt/variables.h>
#include <libxslt/xsltconfig.h>

#define Pystylesheet_Get(v) (((v) == Py_None) ? NULL : \
        (((Pystylesheet_Object *)(v))->obj))

typedef struct {
    PyObject_HEAD
    xsltStylesheetPtr obj;
} Pystylesheet_Object;

#define PytransformCtxt_Get(v) (((v) == Py_None) ? NULL : \
        (((PytransformCtxt_Object *)(v))->obj))

typedef struct {
    PyObject_HEAD
    xsltTransformContextPtr obj;
} PytransformCtxt_Object;

#define PycompiledStyle_Get(v) (((v) == Py_None) ? NULL : \
        (((PycompiledStyle_Object *)(v))->obj))

typedef struct {
    PyObject_HEAD
    xsltTransformContextPtr obj;
} PycompiledStyle_Object;


PyObject * libxslt_xsltStylesheetPtrWrap(xsltStylesheetPtr ctxt);
PyObject * libxslt_xsltTransformContextPtrWrap(xsltTransformContextPtr ctxt);
PyObject * libxslt_xsltStylePreCompPtrWrap(xsltStylePreCompPtr comp);
PyObject * libxslt_xsltElemPreCompPtrWrap(xsltElemPreCompPtr comp);
