/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 26, 2022.
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

// Copyright 2020 The Abseil Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef ABSL_STRINGS_INTERNAL_STR_FORMAT_FLOAT_CONVERSION_H_
#define ABSL_STRINGS_INTERNAL_STR_FORMAT_FLOAT_CONVERSION_H_

#include "absl/strings/internal/str_format/extension.h"

namespace absl {
ABSL_NAMESPACE_BEGIN
namespace str_format_internal {

bool ConvertFloatImpl(float v, const FormatConversionSpecImpl &conv,
                      FormatSinkImpl *sink);

bool ConvertFloatImpl(double v, const FormatConversionSpecImpl &conv,
                      FormatSinkImpl *sink);

bool ConvertFloatImpl(long double v, const FormatConversionSpecImpl &conv,
                      FormatSinkImpl *sink);

}  // namespace str_format_internal
ABSL_NAMESPACE_END
}  // namespace absl

#endif  // ABSL_STRINGS_INTERNAL_STR_FORMAT_FLOAT_CONVERSION_H_
