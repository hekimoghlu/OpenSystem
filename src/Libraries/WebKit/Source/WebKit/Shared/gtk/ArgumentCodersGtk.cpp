/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 10, 2023.
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
#include "ArgumentCodersGtk.h"

#include "ArgumentCodersGLib.h"
#include <WebCore/GraphicsContext.h>
#include <WebCore/Image.h>
#include <WebCore/SelectionData.h>
#include <WebCore/ShareableBitmap.h>
#include <gtk/gtk.h>
#include <wtf/glib/GUniquePtr.h>

namespace IPC {
using namespace WebCore;
using namespace WebKit;

void ArgumentCoder<GRefPtr<GtkPrintSettings>>::encode(Encoder& encoder, const GRefPtr<GtkPrintSettings>& argument)
{
    GRefPtr<GtkPrintSettings> printSettings = argument ? argument : adoptGRef(gtk_print_settings_new());
    GRefPtr<GVariant> variant = adoptGRef(gtk_print_settings_to_gvariant(printSettings.get()));
    encoder << variant;
}

std::optional<GRefPtr<GtkPrintSettings>> ArgumentCoder<GRefPtr<GtkPrintSettings>>::decode(Decoder& decoder)
{
    auto variant = decoder.decode<GRefPtr<GVariant>>();
    if (UNLIKELY(!variant))
        return std::nullopt;

    return gtk_print_settings_new_from_gvariant(variant->get());
}

void ArgumentCoder<GRefPtr<GtkPageSetup>>::encode(Encoder& encoder, const GRefPtr<GtkPageSetup>& argument)
{
    GRefPtr<GtkPageSetup> pageSetup = argument ? argument : adoptGRef(gtk_page_setup_new());
    GRefPtr<GVariant> variant = adoptGRef(gtk_page_setup_to_gvariant(pageSetup.get()));
    encoder << variant;
}

std::optional<GRefPtr<GtkPageSetup>> ArgumentCoder<GRefPtr<GtkPageSetup>>::decode(Decoder& decoder)
{
    auto variant = decoder.decode<GRefPtr<GVariant>>();
    if (UNLIKELY(!variant))
        return std::nullopt;

    return gtk_page_setup_new_from_gvariant(variant->get());
}

}
