/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 14, 2025.
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
#include "DIBPixelData.h"

namespace WebCore {

#ifndef NDEBUG
static const WORD bitmapType = 0x4d42; // BMP format
static const WORD bitmapPixelsPerMeter = 2834; // 72 dpi
#endif

DIBPixelData::DIBPixelData(HBITMAP bitmap)
{
    initialize(bitmap);
}

void DIBPixelData::initialize(HBITMAP bitmap)
{
    BITMAP bmpInfo = {0, 0, 0, 0, 0, 0, nullptr};
    GetObject(bitmap, sizeof(bmpInfo), &bmpInfo);

    m_bitmapBuffer = reinterpret_cast<UInt8*>(bmpInfo.bmBits);
    m_bitmapBufferLength = bmpInfo.bmWidthBytes * bmpInfo.bmHeight;
    m_size = IntSize(bmpInfo.bmWidth, bmpInfo.bmHeight);
    m_bytesPerRow = bmpInfo.bmWidthBytes;
    m_bitsPerPixel = bmpInfo.bmBitsPixel;
}

DIBPixelData::DIBPixelData(void* data, IntSize size)
    : m_bitmapBuffer(reinterpret_cast<UInt8*>(data))
    , m_bitmapBufferLength(8 * size.width() * size.height())
    , m_size(size)
    , m_bytesPerRow(8 * size.width())
    , m_bitsPerPixel(8)
{
}

#ifndef NDEBUG
void DIBPixelData::writeToFile(LPCWSTR filePath)
{
    HANDLE hFile = ::CreateFile(filePath, GENERIC_WRITE, 0, 0, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, 0);
    if (INVALID_HANDLE_VALUE == hFile)
        return;

    BITMAPFILEHEADER header;
    header.bfType = bitmapType;
    header.bfOffBits = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER);
    header.bfReserved1 = 0;
    header.bfReserved2 = 0;
    header.bfSize = sizeof(BITMAPFILEHEADER);

    BITMAPINFOHEADER info;
    info.biSize = sizeof(BITMAPINFOHEADER);
    info.biWidth = m_size.width();
    info.biHeight = m_size.height();
    info.biPlanes = 1;
    info.biBitCount = m_bitsPerPixel;
    info.biCompression = BI_RGB;
    info.biSizeImage = bufferLength();
    info.biXPelsPerMeter = bitmapPixelsPerMeter;
    info.biYPelsPerMeter = bitmapPixelsPerMeter;
    info.biClrUsed = 0;
    info.biClrImportant = 0;

    DWORD bytesWritten = 0;
    ::WriteFile(hFile, &header, sizeof(header), &bytesWritten, 0);
    ::WriteFile(hFile, &info, sizeof(info), &bytesWritten, 0);
    ::WriteFile(hFile, buffer(), bufferLength(), &bytesWritten, 0);

    ::CloseHandle(hFile);
}
#endif

void DIBPixelData::setRGBABitmapAlpha(HDC hdc, const IntRect& dstRect, unsigned char level)
{
    HBITMAP bitmap = static_cast<HBITMAP>(GetCurrentObject(hdc, OBJ_BITMAP));

    if (!bitmap)
        return;

    DIBPixelData pixelData(bitmap);
    ASSERT(pixelData.bitsPerPixel() == 32);

    IntRect drawRect(dstRect);
    XFORM trans;
    GetWorldTransform(hdc, &trans);
    IntSize transformedPosition(trans.eDx, trans.eDy);
    drawRect.move(transformedPosition);

    int pixelDataWidth = pixelData.size().width();
    int pixelDataHeight = pixelData.size().height();
    IntRect bitmapRect(0, 0, pixelDataWidth, pixelDataHeight);
    drawRect.intersect(bitmapRect);
    if (drawRect.isEmpty())
        return;

    RGBQUAD* bytes = reinterpret_cast<RGBQUAD*>(pixelData.buffer());
    bytes += drawRect.y() * pixelDataWidth;

    size_t width = drawRect.width();
    size_t height = drawRect.height();
    int x = drawRect.x();
    for (size_t i = 0; i < height; i++) {
        RGBQUAD* p = bytes + x;
        for (size_t j = 0; j < width; j++) {
            p->rgbReserved = level;
            p++;
        }
        bytes += pixelDataWidth;
    }
}

} // namespace WebCore
