/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 19, 2022.
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

#if ENABLE(UNIFIED_PDF) && PLATFORM(MAC)

#include "PDFDocumentLayout.h"
#include "PDFPluginBase.h"
#include "UnifiedPDFPlugin.h"
#include <PDFKit/PDFKit.h>
#include <wtf/RetainPtr.h>
#include <wtf/ThreadSafeWeakPtr.h>
#include <wtf/WeakObjCPtr.h>

namespace WebCore {
class HTMLPlugInElement;
class WeakPtrImplWithEventTargetData;
}

@interface WKAccessibilityPDFDocumentObject: NSObject {
    RetainPtr<PDFDocument> _pdfDocument;
    WeakObjCPtr<NSObject> _parent;
    ThreadSafeWeakPtr<WebKit::UnifiedPDFPlugin> _pdfPlugin;
}

@property (assign) WeakPtr<WebCore::HTMLPlugInElement, WebCore::WeakPtrImplWithEventTargetData> pluginElement;

- (id)initWithPDFDocument:(RetainPtr<PDFDocument>)document andElement:(WebCore::HTMLPlugInElement*)element;
- (void)setParent:(NSObject *)parent;
- (void)setPDFDocument:(RetainPtr<PDFDocument>)document;
- (void)setPDFPlugin:(WebKit::UnifiedPDFPlugin*)plugin;
- (PDFDocument *)document;
- (NSObject *)accessibilityParent;
- (id)accessibilityHitTest:(NSPoint)point;
- (void)gotoDestination:(PDFDestination *)destination;
- (NSRect)convertFromPDFPageToScreenForAccessibility:(NSRect)rectInPageCoordinate pageIndex:(WebKit::PDFDocumentLayout::PageIndex)pageIndex;
- (id)accessibilityAssociatedControlForAnnotation:(PDFAnnotation *)annotation;
- (void)setActiveAnnotation:(PDFAnnotation *)annotation;

@end

#endif
