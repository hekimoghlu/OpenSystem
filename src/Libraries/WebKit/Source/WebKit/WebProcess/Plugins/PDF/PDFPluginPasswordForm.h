/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 26, 2022.
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

#include "PDFPluginAnnotation.h"

namespace WebKit {

class PDFPluginPasswordForm : public PDFPluginAnnotation {
    WTF_MAKE_FAST_ALLOCATED;
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(PDFPluginPasswordForm);
public:
    static Ref<PDFPluginPasswordForm> create(PDFPluginBase*);
    virtual ~PDFPluginPasswordForm();

    void updateGeometry() override;

    void unlockFailed();

private:
    PDFPluginPasswordForm(PDFPluginBase* plugin)
        : PDFPluginAnnotation(nullptr, plugin)
    {
    }

    Ref<WebCore::Element> createAnnotationElement() override;

    RefPtr<WebCore::Element> m_subtitleElement;
};

} // namespace WebKit

#endif // ENABLE(PDF_PLUGIN)
