/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 31, 2021.
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
#include "DragImage.h"

#include "Element.h"
#include "FloatRoundedRect.h"
#include "FontCascade.h"
#include "FontDescription.h"
#include "FontSelector.h"
#include "HWndDC.h"
#include "Image.h"
#include "StringTruncator.h"
#include "TextIndicator.h"
#include "TextRun.h"
#include "WebCoreTextRenderer.h"
#include <wtf/RetainPtr.h>
#include <wtf/URL.h>
#include <wtf/win/GDIObject.h>

#include <windows.h>

#if USE(CAIRO)
#include "GraphicsContextCairo.h"
#endif

#if USE(SKIA)
#include "GraphicsContextSkia.h"
#endif

namespace WebCore {

#if USE(CAIRO)
GDIObject<HBITMAP> allocImage(HDC, IntSize, PlatformGraphicsContext** targetRef);
void deallocContext(PlatformGraphicsContext* target);
#endif

IntSize dragImageSize(DragImageRef image)
{
    if (!image)
        return IntSize();
    BITMAP b;
    GetObject(image, sizeof(BITMAP), &b);
    return IntSize(b.bmWidth, b.bmHeight);
}

void deleteDragImage(DragImageRef image)
{
    if (image)
        ::DeleteObject(image);
}

DragImageRef dissolveDragImageToFraction(DragImageRef image, float)
{
    //We don't do this on windows as the dragimage is blended by the OS
    return image;
}
        
DragImageRef createDragImageIconForCachedImageFilename(const String& filename)
{
    SHFILEINFO shfi { };
    auto fname = filename.wideCharacters();
    if (FAILED(SHGetFileInfo(fname.data(), FILE_ATTRIBUTE_NORMAL, &shfi, sizeof(shfi), SHGFI_ICON | SHGFI_USEFILEATTRIBUTES)))
        return 0;

    ICONINFO iconInfo;
    if (!GetIconInfo(shfi.hIcon, &iconInfo)) {
        DestroyIcon(shfi.hIcon);
        return 0;
    }

    DestroyIcon(shfi.hIcon);
    DeleteObject(iconInfo.hbmMask);

    return iconInfo.hbmColor;
}

#if USE(CAIRO)
const float DragLabelBorderX = 4;
// Keep border_y in synch with DragController::LinkDragBorderInset.
const float DragLabelBorderY = 2;
const float DragLabelRadius = 5;
const float LabelBorderYOffset = 2;

const float MaxDragLabelWidth = 200;
const float MaxDragLabelStringWidth = (MaxDragLabelWidth - 2 * DragLabelBorderX);

const float DragLinkLabelFontsize = 11;
const float DragLinkUrlFontSize = 10;

static FontCascade dragLabelFont(int size, bool bold)
{
    FontCascade result;
    NONCLIENTMETRICS metrics;
    metrics.cbSize = sizeof(metrics);
    SystemParametersInfo(SPI_GETNONCLIENTMETRICS, metrics.cbSize, &metrics, 0);

    FontCascadeDescription description;
    description.setWeight(bold ? boldWeightValue() : normalWeightValue());
    description.setOneFamily(metrics.lfSmCaptionFont.lfFaceName);
    description.setSpecifiedSize((float)size);
    description.setComputedSize((float)size);
    result = FontCascade(WTFMove(description));
    result.update();
    return result;
}

DragImageRef createDragImageForLink(Element&, URL& url, const String& inLabel, TextIndicatorData&, float)
{
    static const FontCascade labelFont = dragLabelFont(DragLinkLabelFontsize, true);
    static const FontCascade urlFont = dragLabelFont(DragLinkUrlFontSize, false);

    bool drawURLString = true;
    bool clipURLString = false;
    bool clipLabelString = false;

    String urlString = url.string(); 
    String label = inLabel;
    if (label.isEmpty()) {
        drawURLString = false;
        label = urlString;
    }

    // First step in drawing the link drag image width.
    TextRun labelRun(label);
    TextRun urlRun(urlString);
    IntSize labelSize(labelFont.width(labelRun), labelFont.metricsOfPrimaryFont().intAscent() + labelFont.metricsOfPrimaryFont().intDescent());

    if (labelSize.width() > MaxDragLabelStringWidth) {
        labelSize.setWidth(MaxDragLabelStringWidth);
        clipLabelString = true;
    }
    
    IntSize urlStringSize;
    IntSize imageSize(labelSize.width() + DragLabelBorderX * 2, labelSize.height() + DragLabelBorderY * 2);

    if (drawURLString) {
        urlStringSize.setWidth(urlFont.width(urlRun));
        urlStringSize.setHeight(urlFont.metricsOfPrimaryFont().intAscent() + urlFont.metricsOfPrimaryFont().intDescent());
        imageSize.setHeight(imageSize.height() + urlStringSize.height());
        if (urlStringSize.width() > MaxDragLabelStringWidth) {
            imageSize.setWidth(MaxDragLabelWidth);
            clipURLString = true;
        } else
            imageSize.setWidth(std::max(labelSize.width(), urlStringSize.width()) + DragLabelBorderX * 2);
    }

    // We now know how big the image needs to be, so we create and
    // fill the background
    HWndDC dc(0);
    auto workingDC = adoptGDIObject(::CreateCompatibleDC(dc));
    if (!workingDC)
        return 0;

    PlatformGraphicsContext* contextRef;
    auto image = allocImage(workingDC.get(), imageSize, &contextRef);
    if (!image)
        return 0;
        
    ::SelectObject(workingDC.get(), image.get());
    GraphicsContextCairo context(contextRef);
    // On Mac alpha is {0.7, 0.7, 0.7, 0.8}, however we can't control alpha
    // for drag images on win, so we use 1
    constexpr auto backgroundColor = SRGBA<uint8_t> { 140, 140, 140 };
    static const IntSize radii(DragLabelRadius, DragLabelRadius);
    IntRect rect(0, 0, imageSize.width(), imageSize.height());
    context.fillRoundedRect(FloatRoundedRect(rect, radii, radii, radii, radii), backgroundColor);
 
    // Draw the text
    constexpr auto topColor = Color::black; // original alpha = 0.75
    constexpr auto bottomColor = Color::white.colorWithAlphaByte(127); // original alpha = 0.5
    if (drawURLString) {
        if (clipURLString)
            urlString = StringTruncator::rightTruncate(urlString, imageSize.width() - (DragLabelBorderX * 2.0f), urlFont);
        IntPoint textPos(DragLabelBorderX, imageSize.height() - (LabelBorderYOffset + urlFont.metricsOfPrimaryFont().intDescent()));
        WebCoreDrawDoubledTextAtPoint(context, urlString, textPos, urlFont, topColor, bottomColor);
    }
    
    if (clipLabelString)
        label = StringTruncator::rightTruncate(label, imageSize.width() - (DragLabelBorderX * 2.0f), labelFont);

    IntPoint textPos(DragLabelBorderX, DragLabelBorderY + labelFont.size());
    WebCoreDrawDoubledTextAtPoint(context, label, textPos, labelFont, topColor, bottomColor);

    deallocContext(contextRef);
    return image.leak();
}
#else
DragImageRef createDragImageForLink(Element&, URL&, const String&, TextIndicatorData&, float)
{
    return nullptr;
}
#endif

DragImageRef createDragImageForColor(const Color&, const FloatRect&, float, Path&)
{
    return nullptr;
}

#if USE(SKIA)
DragImageRef createDragImageFromImage(Image*, ImageOrientation)
{
    return nullptr;
}

DragImageRef scaleDragImage(DragImageRef, FloatSize)
{
    return nullptr;
}
#endif

}
