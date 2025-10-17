/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 29, 2022.
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
#ifndef __AGG_PY_TRANSFORMS_H__
#define __AGG_PY_TRANSFORMS_H__

#include "agg_trans_affine.h"

/** A helper function to convert from a Numpy affine transformation matrix
 *  to an agg::trans_affine.
 */
agg::trans_affine
py_to_agg_transformation_matrix(PyObject* obj, bool errors = true);

bool
py_convert_bbox(PyObject* bbox_obj, double& l, double& b, double& r, double& t);

#endif // __AGG_PY_TRANSFORMS_H__
