/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 6, 2022.
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

#include "BMPImageReader.h"

namespace WebCore {

// This class decodes the ICO and CUR image formats.
class ICOImageDecoder final : public ScalableImageDecoder {
public:
    static Ref<ScalableImageDecoder> create(AlphaOption alphaOption, GammaAndColorProfileOption gammaAndColorProfileOption)
    {
        return adoptRef(*new ICOImageDecoder(alphaOption, gammaAndColorProfileOption));
    }

    virtual ~ICOImageDecoder();

    // ScalableImageDecoder
    String filenameExtension() const final { return "ico"_s; }
    void setData(const FragmentedSharedBuffer&, bool allDataReceived) final;
    IntSize size() const final;
    IntSize frameSizeAtIndex(size_t, SubsamplingLevel) const final;
    bool setSize(const IntSize&) final;
    size_t frameCount() const final;
    ScalableImageDecoderFrame* frameBufferAtIndex(size_t) final;
    // CAUTION: setFailed() deletes all readers and decoders. Be careful to
    // avoid accessing deleted memory, especially when calling this from
    // inside BMPImageReader!
    bool setFailed() final;
    std::optional<IntPoint> hotSpot() const final;

private:
    enum ImageType {
        Unknown,
        BMP,
        PNG,
    };

    enum FileType {
        ICON = 1,
        CURSOR = 2,
    };

    struct IconDirectoryEntry {
        IntSize m_size;
        uint16_t m_bitCount;
        IntPoint m_hotSpot;
        uint32_t m_imageOffset;
    };

    ICOImageDecoder(AlphaOption, GammaAndColorProfileOption);
    void tryDecodeSize(bool allDataReceived) final { decode(0, true, allDataReceived); }

    // Returns true if |a| is a preferable icon entry to |b|.
    // Larger sizes, or greater bitdepths at the same size, are preferable.
    static bool compareEntries(const IconDirectoryEntry& a, const IconDirectoryEntry& b);

    inline uint16_t readUint16(int offset) const
    {
        return BMPImageReader::readUint16(*m_data, m_decodedOffset + offset);
    }

    inline uint32_t readUint32(int offset) const
    {
        return BMPImageReader::readUint32(*m_data, m_decodedOffset + offset);
    }

    // If the desired PNGImageDecoder exists, gives it the appropriate data.
    void setDataForPNGDecoderAtIndex(size_t);

    // Decodes the entry at |index|. If |onlySize| is true, stops decoding
    // after calculating the image size. If decoding fails but there is no
    // more data coming, sets the "decode failure" flag.
    void decode(size_t index, bool onlySize, bool allDataReceived);

    // Decodes the directory and directory entries at the beginning of the
    // data, and initializes members. Returns true if all decoding
    // succeeded. Once this returns true, all entries' sizes are known.
    bool decodeDirectory();

    // Decodes the specified entry.
    bool decodeAtIndex(size_t);

    // Processes the ICONDIR at the beginning of the data. Returns true if
    // the directory could be decoded.
    bool processDirectory();

    // Processes the ICONDIRENTRY records after the directory. Keeps the
    // "best" entry as the one we'll decode. Returns true if the entries
    // could be decoded.
    bool processDirectoryEntries();

    // Returns the hot-spot for |index|, returns std::nullopt if there is none.
    std::optional<IntPoint> hotSpotAtIndex(size_t) const;

    // Reads and returns a directory entry from the current offset into
    // |data|.
    IconDirectoryEntry readDirectoryEntry();

    // Determines whether the desired entry is a BMP or PNG. Returns true
    // if the type could be determined.
    ImageType imageTypeAtIndex(size_t);

    // An index into |m_data| representing how much we've already decoded.
    // Note that this only tracks data _this_ class decodes; once the
    // BMPImageReader takes over this will not be updated further.
    size_t m_decodedOffset;

    // Which type of file (ICO/CUR) this is.
    FileType m_fileType;

    // The headers for the ICO.
    typedef Vector<IconDirectoryEntry> IconDirectoryEntries;
    IconDirectoryEntries m_dirEntries;

    // The image decoders for the various frames.
    typedef Vector<std::unique_ptr<BMPImageReader>> BMPReaders;
    BMPReaders m_bmpReaders;
    typedef Vector<RefPtr<ScalableImageDecoder>> PNGDecoders;
    PNGDecoders m_pngDecoders;

    // Valid only while a BMPImageReader is decoding, this holds the size
    // for the particular entry being decoded.
    IntSize m_frameSize;
};

} // namespace WebCore
