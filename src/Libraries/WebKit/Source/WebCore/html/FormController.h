/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 25, 2025.
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
#include <wtf/HashMap.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class Document;
class FormListedElement;
class HTMLFormElement;
class ValidatedFormListedElement;

using FormControlState = Vector<AtomString>;

class FormController {
    WTF_MAKE_TZONE_ALLOCATED(FormController);

public:
    FormController();
    ~FormController();

    WEBCORE_EXPORT Vector<AtomString> formElementsState(const Document&) const;
    WEBCORE_EXPORT void setStateForNewFormElements(const Vector<AtomString>& stateVector);

    void willDeleteForm(HTMLFormElement&);
    void restoreControlStateFor(ValidatedFormListedElement&);
    void restoreControlStateIn(HTMLFormElement&);
    bool hasFormStateToRestore() const;

    WEBCORE_EXPORT static Vector<String> referencedFilePaths(const Vector<AtomString>& stateVector);
    static HTMLFormElement* ownerForm(const FormListedElement&);

private:
    class FormKeyGenerator;
    class SavedFormState;
    using SavedFormStateMap = UncheckedKeyHashMap<String, SavedFormState>;

    FormControlState takeStateForFormElement(const ValidatedFormListedElement&);
    static SavedFormStateMap parseStateVector(const Vector<AtomString>&);

    SavedFormStateMap m_savedFormStateMap;
    std::unique_ptr<FormKeyGenerator> m_formKeyGenerator;
};

} // namespace WebCore
