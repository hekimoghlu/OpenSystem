/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 21, 2024.
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

#include "WebDateTimePicker.h"
#include <WebCore/DateComponents.h>
#include <WebCore/DateTimeChooserParameters.h>

namespace WebKit {

class WebDateTimePickerGtk final : public WebDateTimePicker {
public:
    static Ref<WebDateTimePickerGtk> create(WebPageProxy&);
    ~WebDateTimePickerGtk();

private:
    WebDateTimePickerGtk(WebPageProxy&);

    void endPicker() final;
    void showDateTimePicker(WebCore::DateTimeChooserParameters&&) final;

    void update(WebCore::DateTimeChooserParameters&&);
    void didChooseDate();
    void invalidate();

    GtkWidget* m_popover { nullptr };
    GtkWidget* m_calendar { nullptr };
    std::optional<WebCore::DateComponents> m_currentDate;
    WebCore::SecondFormat m_secondFormat { WebCore::SecondFormat::None };
    bool m_inUpdate { false };
};

} // namespace WebKit
