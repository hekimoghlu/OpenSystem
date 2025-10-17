/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 26, 2024.
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
#ifndef Icon_h
#define Icon_h

#include <wtf/Forward.h>
#include <wtf/RefCounted.h>
#include <wtf/RefPtr.h>
#include <wtf/RetainPtr.h>

#if PLATFORM(COCOA)
#include "NativeImage.h"
#include "PlatformImage.h"
#include <CoreGraphics/CoreGraphics.h>

#if USE(APPKIT)
OBJC_CLASS NSImage;
using CocoaImage = NSImage;
#else
OBJC_CLASS UIImage;
using CocoaImage = UIImage;
#endif

#elif PLATFORM(WIN)
typedef struct HICON__* HICON;

#elif PLATFORM(GTK)
#include <wtf/glib/GRefPtr.h>

typedef struct _GIcon GIcon;
#endif

namespace WebCore {

class GraphicsContext;
class FloatRect;
class NativeImage;

class Icon : public RefCounted<Icon> {
public:
    WEBCORE_EXPORT static RefPtr<Icon> createIconForFiles(const Vector<String>& filenames);

    WEBCORE_EXPORT ~Icon();

    void paint(GraphicsContext&, const FloatRect&);

#if PLATFORM(WIN)
    static Ref<Icon> create(HICON hIcon) { return adoptRef(*new Icon(hIcon)); }
#endif

#if PLATFORM(GTK)
    WEBCORE_EXPORT static RefPtr<Icon> create(GIcon*);

    GIcon* icon() const { return m_icon.get(); };
#endif

#if PLATFORM(COCOA)
    WEBCORE_EXPORT static RefPtr<Icon> create(CocoaImage *);
    WEBCORE_EXPORT static RefPtr<Icon> create(PlatformImagePtr&&);

    RetainPtr<CocoaImage> image() const { return m_image; };
#endif

#if PLATFORM(MAC)
    static RefPtr<Icon> createIconForUTI(const String&);
    static RefPtr<Icon> createIconForFileExtension(const String&);
#endif

private:
#if PLATFORM(COCOA)
    Icon(CocoaImage *);
    RetainPtr<CocoaImage> m_image;
#elif PLATFORM(WIN)
    Icon(HICON);
    HICON m_hIcon;
#elif PLATFORM(GTK)
    explicit Icon(GIcon*);
    GRefPtr<GIcon> m_icon;
#endif
};

}

#endif
