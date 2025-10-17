/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 13, 2025.
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
#pragma once

#include <wtf/Forward.h>

namespace WebCore {

class CSSToLengthConversionData;
class CSSValue;
class RotateTransformOperation;
class ScaleTransformOperation;
class TransformOperation;
class TransformOperations;
class TranslateTransformOperation;

namespace Style {

Ref<TransformOperation> createTransformOperation(const CSSValue&, const CSSToLengthConversionData&);
TransformOperations createTransformOperations(const CSSValue&, const CSSToLengthConversionData&);

RefPtr<TranslateTransformOperation> createTranslate(const CSSValue&, const CSSToLengthConversionData&);
RefPtr<RotateTransformOperation> createRotate(const CSSValue&, const CSSToLengthConversionData&);
RefPtr<ScaleTransformOperation> createScale(const CSSValue&, const CSSToLengthConversionData&);

} // namespace Style
} // namespace WebCore
