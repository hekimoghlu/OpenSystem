/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 19, 2025.
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
#import "config.h"

#if HAVE(PDFKIT)

#import "PDFKitSPI.h"
#import <wtf/SoftLinking.h>

SOFT_LINK_FRAMEWORK_FOR_SOURCE(WebKit, PDFKit)

#if PLATFORM(APPLETV)
SOFT_LINK_CLASS_FOR_SOURCE(WebKit, PDFKit, PDFHostViewController)
#endif

SOFT_LINK_CLASS_FOR_SOURCE(WebKit, PDFKit, PDFActionResetForm)
SOFT_LINK_CLASS_FOR_SOURCE(WebKit, PDFKit, PDFDocument)
SOFT_LINK_CLASS_FOR_SOURCE(WebKit, PDFKit, PDFLayerController)
SOFT_LINK_CLASS_FOR_SOURCE(WebKit, PDFKit, PDFSelection)
SOFT_LINK_CONSTANT_FOR_SOURCE(WebKit, PDFKit, kPDFDestinationUnspecifiedValue, CGFloat)
SOFT_LINK_CONSTANT_FOR_SOURCE(WebKit, PDFKit, PDFViewCopyPermissionNotification, NSNotificationName)
SOFT_LINK_CONSTANT_FOR_SOURCE(WebKit, PDFKit, PDFDocumentCreationDateAttribute, PDFDocumentAttribute)
SOFT_LINK_CONSTANT_FOR_SOURCE(WebKit, PDFKit, PDFAnnotationKeySubtype, NSString *)
SOFT_LINK_CONSTANT_FOR_SOURCE(WebKit, PDFKit, PDFAnnotationKeyWidgetFieldType, NSString *)
SOFT_LINK_CONSTANT_FOR_SOURCE(WebKit, PDFKit, PDFAnnotationSubtypeLink, NSString *)
SOFT_LINK_CONSTANT_FOR_SOURCE(WebKit, PDFKit, PDFAnnotationSubtypePopup, NSString *)
SOFT_LINK_CONSTANT_FOR_SOURCE(WebKit, PDFKit, PDFAnnotationSubtypeText, NSString *)
SOFT_LINK_CONSTANT_FOR_SOURCE(WebKit, PDFKit, PDFAnnotationSubtypeWidget, NSString *)
SOFT_LINK_CONSTANT_FOR_SOURCE(WebKit, PDFKit, PDFAnnotationWidgetSubtypeButton, NSString *)
SOFT_LINK_CONSTANT_FOR_SOURCE(WebKit, PDFKit, PDFAnnotationWidgetSubtypeChoice, NSString *)
SOFT_LINK_CONSTANT_FOR_SOURCE(WebKit, PDFKit, PDFAnnotationWidgetSubtypeSignature, NSString *)
SOFT_LINK_CONSTANT_FOR_SOURCE(WebKit, PDFKit, PDFAnnotationWidgetSubtypeText, NSString *)

#endif // HAVE(PDFKIT)
