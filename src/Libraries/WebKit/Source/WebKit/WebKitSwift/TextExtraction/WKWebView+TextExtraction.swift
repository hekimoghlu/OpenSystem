/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 12, 2024.
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

// Copyright (C) 2024 Apple Inc. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
// 1. Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY APPLE INC. AND ITS CONTRIBUTORS ``AS IS''
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
// THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL APPLE INC. OR ITS CONTRIBUTORS
// BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
// THE POSSIBILITY OF SUCH DAMAGE.

#if canImport(UIIntelligenceSupport)

@_spiOnly import WebKit
@_spiOnly import UIIntelligenceSupport

#if canImport(UIKit)
@_spi(UIIntelligenceSupport) import UIKit
#else
@_spi(UIIntelligenceSupport) import AppKit
#endif

private func createEditable(for editable: WKTextExtractionEditable?) -> IntelligenceElement.Text.Editable? {
    guard let editable else { return .none }
    return .init(label: editable.label, prompt: editable.placeholder, contentType: nil, isSecure: editable.isSecure, isFocused: editable.isFocused)
}

private func createElementContent(for item: WKTextExtractionItem) -> IntelligenceElement.Content {
    switch item {
    case let text as WKTextExtractionTextItem:
        var content = AttributedString(text.content)
        if text.selectedRange.location != NSNotFound {
            if let range = Range(text.selectedRange, in: content) {
                content[range].intelligenceSelected = true
            }
        }
        for link in text.links {
            if let range = Range(link.range, in: content) {
                content[range].intelligenceLink = link.url as URL
            }
        }
        return .text(IntelligenceElement.Text(attributedText: content, editable: createEditable(for: text.editable)))
    case let image as WKTextExtractionImageItem:
        return .image(IntelligenceElement.Image(name: image.name, textDescription: image.altText))
    default:
        return .base
    }
}

private func createIntelligenceElement(item: WKTextExtractionItem) -> IntelligenceElement {
    var element = IntelligenceElement(boundingBox: item.rectInWebView, content: createElementContent(for: item))
    element.subelements = item.children.map { child in createIntelligenceElement(item: child) }
    return element
}

@_spi(WKIntelligenceSupport)
extension WKWebView {
    open override var _intelligenceBaseClass: AnyClass {
        WKWebView.self
    }

    open override func _intelligenceCollectContent(in visibleRect: CGRect, collector: UIIntelligenceElementCollector) {
#if canImport(UIIntelligenceSupport, _version: 9007)
        let context = collector.context.createRemoteContext(description: "WKWebView")
#else
        let context = collector.context.createRemoteContext()
#endif
        collector.collect(.remote(context))
    }

    open override func _intelligenceCollectRemoteContent(in visibleRect: CGRect, remoteContextWrapper: UIIntelligenceCollectionRemoteContextWrapper) {
        let coordinator = IntelligenceCollectionCoordinator.shared
        coordinator.createCollector(remoteContextWrapper: remoteContextWrapper) { collector, error in
            guard let collector else { return }
            self._requestTextExtraction(in: visibleRect) { item in
                if let item {
                    collector.collect(createIntelligenceElement(item: item))
                }
                coordinator.finishCollection(collector)
            }
        }
    }

    @objc private func _requestTextExtraction(in rect: CGRect, completionHandler: @escaping (WKTextExtractionItem?) -> Void) {
        let context = WKTextExtractionRequest(rectInWebView: rect, completionHandler)
        self.perform(Selector(("_requestTextExtractionForSwift:")), with: context)
    }
}

#endif // canImport(UIIntelligenceSupport)
