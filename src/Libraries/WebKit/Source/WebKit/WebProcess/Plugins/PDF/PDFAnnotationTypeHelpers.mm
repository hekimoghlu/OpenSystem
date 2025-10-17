/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 28, 2023.
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
#include "config.h"
#include "PDFAnnotationTypeHelpers.h"

#if ENABLE(PDF_PLUGIN)

#include "PDFKitSPI.h"
#include <optional>

#include "PDFKitSoftLink.h"

namespace WebKit::PDFAnnotationTypeHelpers {

static std::optional<WidgetType> widgetType(PDFAnnotation *annotation)
{
    if (!annotationIsOfType(annotation, AnnotationType::Widget))
        return { };

    NSString *type = [annotation valueForAnnotationKey:get_PDFKit_PDFAnnotationKeyWidgetFieldType()];
    if ([type isEqualToString:get_PDFKit_PDFAnnotationWidgetSubtypeButton()])
        return WidgetType::Button;
    if ([type isEqualToString:get_PDFKit_PDFAnnotationWidgetSubtypeChoice()])
        return WidgetType::Choice;
    if ([type isEqualToString:get_PDFKit_PDFAnnotationWidgetSubtypeSignature()])
        return WidgetType::Signature;
    if ([type isEqualToString:get_PDFKit_PDFAnnotationWidgetSubtypeText()])
        return WidgetType::Text;

    ASSERT_NOT_REACHED();
    return { };
}

static std::optional<AnnotationType> annotationType(PDFAnnotation *annotation)
{
    NSString *type = [annotation valueForAnnotationKey:get_PDFKit_PDFAnnotationKeySubtype()];
    if ([type isEqualToString:get_PDFKit_PDFAnnotationSubtypeLink()])
        return AnnotationType::Link;
    if ([type isEqualToString:get_PDFKit_PDFAnnotationSubtypePopup()])
        return AnnotationType::Popup;
    if ([type isEqualToString:get_PDFKit_PDFAnnotationSubtypeText()])
        return AnnotationType::Text;
    if ([type isEqualToString:get_PDFKit_PDFAnnotationSubtypeWidget()])
        return AnnotationType::Widget;

    ASSERT_NOT_REACHED();
    return { };
}

template <typename Type>
using AnnotationToTypeConverter = std::optional<Type> (*)(PDFAnnotation *);

template <typename Type>
bool annotationCheckerInternal(PDFAnnotation *annotation, Type type, AnnotationToTypeConverter<Type> converter)
{
    return converter(annotation).transform([queryType = type](Type candidateType) {
        return candidateType == queryType;
    }).value_or(false);
}

template <typename Type>
bool annotationCheckerInternal(PDFAnnotation *annotation, std::initializer_list<Type>&& types, AnnotationToTypeConverter<Type> converter)
{
    auto checker = [annotation, converter = WTFMove(converter)](auto&& type) {
        return annotationCheckerInternal(annotation, std::forward<decltype(type)>(type), WTFMove(converter));
    };
    ASSERT(std::ranges::count_if(types, checker) <= 1);
    return std::ranges::any_of(WTFMove(types), WTFMove(checker));
}

bool annotationIsOfType(PDFAnnotation *annotation, AnnotationType type)
{
    return annotationCheckerInternal(annotation, type, annotationType);
}

bool annotationIsOfType(PDFAnnotation *annotation, std::initializer_list<AnnotationType>&& types)
{
    return annotationCheckerInternal(annotation, WTFMove(types), annotationType);
}

bool annotationIsWidgetOfType(PDFAnnotation *annotation, WidgetType type)
{
    return annotationCheckerInternal(annotation, type, widgetType);
}

bool annotationIsWidgetOfType(PDFAnnotation *annotation, std::initializer_list<WidgetType>&& types)
{
    return annotationCheckerInternal(annotation, WTFMove(types), widgetType);
}

} // namespace WebKit::PDFAnnotationTypeHelpers

#endif // ENABLE(PDF_PLUGIN)
