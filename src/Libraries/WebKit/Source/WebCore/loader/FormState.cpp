/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 23, 2022.
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
#include "FormState.h"
#include "FrameDestructionObserverInlines.h"

#include "Document.h"
#include "HTMLFormElement.h"

namespace WebCore {

inline FormState::FormState(HTMLFormElement& form, StringPairVector&& textFieldValues, Document& sourceDocument, FormSubmissionTrigger formSubmissionTrigger)
    : FrameDestructionObserver(sourceDocument.frame())
    , m_form(form)
    , m_textFieldValues(WTFMove(textFieldValues))
    , m_sourceDocument(sourceDocument)
    , m_formSubmissionTrigger(formSubmissionTrigger)
{
    RELEASE_ASSERT(sourceDocument.frame());
}

FormState::~FormState() = default;

Ref<FormState> FormState::create(HTMLFormElement& form, StringPairVector&& textFieldValues, Document& sourceDocument, FormSubmissionTrigger formSubmissionTrigger)
{
    return adoptRef(*new FormState(form, WTFMove(textFieldValues), sourceDocument, formSubmissionTrigger));
}

void FormState::willDetachPage()
{
    // Beartrap for <rdar://problem/37579354>
    RELEASE_ASSERT(refCount());
}

Ref<Document> FormState::protectedSourceDocument() const
{
    return m_sourceDocument;
}

}
