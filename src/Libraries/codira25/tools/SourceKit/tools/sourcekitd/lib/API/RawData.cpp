/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 21, 2023.
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

//===--- RawData.cpp ------------------------------------------------------===//
//
// Copyright (c) NeXTHub Corporation. All rights reserved.
// DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
//
// This code is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
// version 2 for more details (a copy is included in the LICENSE file that
// accompanied this code).
//
// Author(-s): Tunjay Akbarli
//

//===----------------------------------------------------------------------===//

#include "sourcekitd/RawData.h"

using namespace sourcekitd;

struct RawDataFuncs {
  static sourcekitd_variant_type_t get_type(sourcekitd_variant_t var) {
    return SOURCEKITD_VARIANT_TYPE_DATA;
  }

  static size_t data_get_size(sourcekitd_variant_t variant) {
    return variant.data[2];
  }

  static const void *data_get_ptr(sourcekitd_variant_t variant) {
    return reinterpret_cast<void *>(variant.data[1]);
  }

  static VariantFunctions Funcs;
};

VariantFunctions RawDataFuncs::Funcs = {
    get_type,
    nullptr /*AnnotArray_array_apply*/,
    nullptr /*AnnotArray_array_get_bool*/,
    nullptr /*AnnotArray_array_get_double*/,
    nullptr /*Annot_array_array_get_count*/,
    nullptr /*AnnotArray_array_get_int64*/,
    nullptr /*AnnotArray_array_get_string*/,
    nullptr /*AnnotArray_array_get_uid*/,
    nullptr /*AnnotArray_array_get_value*/,
    nullptr /*AnnotArray_bool_get_value*/,
    nullptr /*AnnotArray_double_get_value*/,
    nullptr /*AnnotArray_dictionary_apply*/,
    nullptr /*AnnotArray_dictionary_get_bool*/,
    nullptr /*AnnotArray_dictionary_get_double*/,
    nullptr /*AnnotArray_dictionary_get_int64*/,
    nullptr /*AnnotArray_dictionary_get_string*/,
    nullptr /*AnnotArray_dictionary_get_value*/,
    nullptr /*AnnotArray_dictionary_get_uid*/,
    nullptr /*AnnotArray_string_get_length*/,
    nullptr /*AnnotArray_string_get_ptr*/,
    nullptr /*AnnotArray_int64_get_value*/,
    nullptr /*AnnotArray_uid_get_value*/,
    data_get_size,
    data_get_ptr,
};

VariantFunctions *sourcekitd::getVariantFunctionsForRawData() {
  return &RawDataFuncs::Funcs;
}
