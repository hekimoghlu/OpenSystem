/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 17, 2023.
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

#if ENABLE(PDF_PLUGIN)

#include <initializer_list>

OBJC_CLASS PDFAnnotation;

namespace WebKit::PDFAnnotationTypeHelpers {

enum class AnnotationType : uint8_t {
    Link,
    Popup,
    Text,
    Widget,
};

enum class WidgetType : uint8_t {
    Button,
    Choice,
    Signature,
    Text,
};

bool annotationIsOfType(PDFAnnotation *, AnnotationType);
bool annotationIsOfType(PDFAnnotation *, std::initializer_list<AnnotationType>&&);
bool annotationIsWidgetOfType(PDFAnnotation *, WidgetType);
bool annotationIsWidgetOfType(PDFAnnotation *, std::initializer_list<WidgetType>&&);

} // namespace WebKit::PDFAnnotationTypeHelpers

#endif // ENABLE(PDF_PLUGIN)
