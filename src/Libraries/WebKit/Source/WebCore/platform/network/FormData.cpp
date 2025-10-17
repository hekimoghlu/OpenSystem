/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 28, 2022.
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
#include "FormData.h"

#include "BlobRegistryImpl.h"
#include "BlobURL.h"
#include "Chrome.h"
#include "DOMFormData.h"
#include "File.h"
#include "FormDataBuilder.h"
#include "MIMETypeRegistry.h"
#include "Page.h"
#include "SharedBuffer.h"
#include <pal/text/TextEncoding.h>
#include "ThreadableBlobRegistry.h"
#include <wtf/FileSystem.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/LineEnding.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(FormData);

inline FormData::FormData(const FormData& data)
    : RefCounted<FormData>()
    , m_elements(data.m_elements)
    , m_identifier(data.m_identifier)
{
}

FormData::~FormData() = default;

Ref<FormData> FormData::create()
{
    return adoptRef(*new FormData);
}

Ref<FormData> FormData::create(std::span<const uint8_t> data)
{
    auto result = create();
    result->appendData(data);
    return result;
}

Ref<FormData> FormData::create(const CString& string)
{
    return create(string.span());
}

Ref<FormData> FormData::create(Vector<uint8_t>&& vector)
{
    auto data = create();
    data->m_elements.append(WTFMove(vector));
    return data;
}

Ref<FormData> FormData::create(const Vector<uint8_t>& vector)
{
    return create(vector.span());
}

Ref<FormData> FormData::create(const DOMFormData& formData, EncodingType encodingType)
{
    auto result = create();
    result->appendNonMultiPartKeyValuePairItems(formData, encodingType);
    return result;
}

Ref<FormData> FormData::create(Vector<WebCore::FormDataElement>&& elements, uint64_t identifier, bool alwaysStream, Vector<uint8_t>&& boundary)
{
    auto result = create();
    result->setAlwaysStream(alwaysStream);
    result->m_boundary = WTFMove(boundary);
    result->m_elements = WTFMove(elements);
    result->setIdentifier(identifier);
    return result;
}

Ref<FormData> FormData::createMultiPart(const DOMFormData& formData)
{
    auto result = create();
    result->appendMultiPartKeyValuePairItems(formData);
    return result;
}

Ref<FormData> FormData::copy() const
{
    return adoptRef(*new FormData(*this));
}

Ref<FormData> FormData::isolatedCopy() const
{
    // FIXME: isolatedCopy() does not copy m_identifier, m_boundary, or m_containsPasswordData.
    // Is all of that correct and intentional?
    auto formData = create();
    formData->m_alwaysStream = m_alwaysStream;
    formData->m_elements = crossThreadCopy(m_elements);
    return formData;
}

unsigned FormData::imageOrMediaFilesCount() const
{
    unsigned imageOrMediaFilesCount = 0;
    for (auto& element : m_elements) {
        auto* encodedFileData = std::get_if<FormDataElement::EncodedFileData>(&element.data);
        if (!encodedFileData)
            continue;

        auto mimeType = MIMETypeRegistry::mimeTypeForPath(encodedFileData->filename);
        if (MIMETypeRegistry::isSupportedImageMIMEType(mimeType) || MIMETypeRegistry::isSupportedMediaMIMEType(mimeType))
            ++imageOrMediaFilesCount;
    }
    return imageOrMediaFilesCount;
}

uint64_t FormDataElement::lengthInBytes(const Function<uint64_t(const URL&)>& blobSize) const
{
    return WTF::switchOn(data,
        [] (const Vector<uint8_t>& bytes) {
            return static_cast<uint64_t>(bytes.size());
        }, [] (const FormDataElement::EncodedFileData& fileData) {
            if (fileData.fileLength != BlobDataItem::toEndOfFile)
                return static_cast<uint64_t>(fileData.fileLength);
            return FileSystem::fileSize(fileData.filename).value_or(0);
        }, [&blobSize] (const FormDataElement::EncodedBlobData& blobData) {
            return blobSize(blobData.url);
        }
    );
}

uint64_t FormDataElement::lengthInBytes() const
{
    return lengthInBytes([](auto& url) {
        return ThreadableBlobRegistry::blobSize(url);
    });
}

FormDataElement FormDataElement::isolatedCopy() const
{
    return WTF::switchOn(data,
        [] (const Vector<uint8_t>& bytes) {
            return FormDataElement(Vector<uint8_t> { bytes });
        }, [] (const FormDataElement::EncodedFileData& fileData) {
            return FormDataElement(fileData.isolatedCopy());
        }, [] (const FormDataElement::EncodedBlobData& blobData) {
            return FormDataElement(blobData.url.isolatedCopy());
        }
    );
}

void FormData::appendData(std::span<const uint8_t> data)
{
    m_lengthInBytes = std::nullopt;
    if (!m_elements.isEmpty()) {
        if (auto* vector = std::get_if<Vector<uint8_t>>(&m_elements.last().data)) {
            vector->append(data);
            return;
        }
    }
    m_elements.append(Vector(data));
}

void FormData::appendFile(const String& filename)
{
    m_elements.append(FormDataElement(filename, 0, BlobDataItem::toEndOfFile, std::nullopt));
    m_lengthInBytes = std::nullopt;
}

void FormData::appendFileRange(const String& filename, long long start, long long length, std::optional<WallTime> expectedModificationTime)
{
    m_elements.append(FormDataElement(filename, start, length, expectedModificationTime));
    m_lengthInBytes = std::nullopt;
}

void FormData::appendBlob(const URL& blobURL)
{
    m_elements.append(FormDataElement(blobURL));
    m_lengthInBytes = std::nullopt;
}

static Vector<uint8_t> normalizeStringData(PAL::TextEncoding& encoding, const String& value)
{
    return normalizeLineEndingsToCRLF(encoding.encode(value, PAL::UnencodableHandling::Entities, PAL::NFCNormalize::No));
}

void FormData::appendMultiPartFileValue(const File& file, Vector<uint8_t>& header, PAL::TextEncoding& encoding)
{
    auto name = file.name();

    // We have to include the filename=".." part in the header, even if the filename is empty
    FormDataBuilder::addFilenameToMultiPartHeader(header, encoding, name);

    // Add the content type if available, or "application/octet-stream" otherwise (RFC 1867).
    auto contentType = file.type();
    if (contentType.isEmpty())
        contentType = "application/octet-stream"_s;
    ASSERT(Blob::isNormalizedContentType(contentType));

    FormDataBuilder::addContentTypeToMultiPartHeader(header, contentType.ascii());

    FormDataBuilder::finishMultiPartHeader(header);
    appendData(header.span());

    if (!file.path().isEmpty())
        appendFile(file.path());
    else if (file.size())
        appendBlob(file.url());
}

void FormData::appendMultiPartStringValue(const String& string, Vector<uint8_t>& header, PAL::TextEncoding& encoding)
{
    FormDataBuilder::finishMultiPartHeader(header);
    appendData(header.span());

    auto normalizedStringData = normalizeStringData(encoding, string);
    appendData(normalizedStringData.span());
}

void FormData::appendMultiPartKeyValuePairItems(const DOMFormData& formData)
{
    m_boundary = FormDataBuilder::generateUniqueBoundaryString();

    auto encoding = formData.encoding();

    Vector<uint8_t> encodedData;
    for (auto& item : formData.items()) {
        auto normalizedName = normalizeStringData(encoding, item.name);
    
        Vector<uint8_t> header;
        FormDataBuilder::beginMultiPartHeader(header, m_boundary.span(), normalizedName);

        if (std::holds_alternative<RefPtr<File>>(item.data))
            appendMultiPartFileValue(*std::get<RefPtr<File>>(item.data), header, encoding);
        else
            appendMultiPartStringValue(std::get<String>(item.data), header, encoding);

        constexpr std::array<uint8_t, 2> newline { '\r', '\n' };
        appendData(newline);
    }
    
    FormDataBuilder::addBoundaryToMultiPartHeader(encodedData, m_boundary.span(), true);

    appendData(encodedData.span());
}

void FormData::appendNonMultiPartKeyValuePairItems(const DOMFormData& formData, EncodingType encodingType)
{
    auto encoding = formData.encoding();

    Vector<uint8_t> encodedData;
    for (auto& item : formData.items()) {
        String stringValue = WTF::switchOn(item.data,
            [](const String& string) {
                return string;
            }, [](const RefPtr<File>& file) {
                return file->name();
            }
        );

        auto normalizedName = normalizeStringData(encoding, item.name);
        auto normalizedStringData = normalizeStringData(encoding, stringValue);
        FormDataBuilder::addKeyValuePairAsFormData(encodedData, normalizedName, normalizedStringData, encodingType);
    }

    appendData(encodedData.span());
}

Vector<uint8_t> FormData::flatten() const
{
    // Concatenate all the byte arrays, but omit any files.
    Vector<uint8_t> data;
    for (auto& element : m_elements) {
        if (auto* vector = std::get_if<Vector<uint8_t>>(&element.data))
            data.append(vector->span());
    }
    return data;
}

String FormData::flattenToString() const
{
    return PAL::Latin1Encoding().decode(flatten().span());
}

static void appendBlobResolved(BlobRegistryImpl* blobRegistry, FormData& formData, const URL& url)
{
    if (!blobRegistry) {
        LOG_ERROR("Tried to resolve a blob without a usable registry");
        return;
    }

    auto* blobData = blobRegistry->getBlobDataFromURL(url);
    if (!blobData) {
        LOG_ERROR("Could not get blob data from a registry");
        return;
    }

    for (const auto& blobItem : blobData->items()) {
        if (blobItem.type() == BlobDataItem::Type::Data) {
            ASSERT(blobItem.data());
            formData.appendData(blobItem.data()->span().subspan(blobItem.offset(), blobItem.length()));
        } else if (blobItem.type() == BlobDataItem::Type::File)
            formData.appendFileRange(blobItem.file()->path(), blobItem.offset(), blobItem.length(), blobItem.file()->expectedModificationTime());
        else
            ASSERT_NOT_REACHED();
    }
}

bool FormData::containsBlobElement() const
{
    for (auto& element : m_elements) {
        if (std::holds_alternative<FormDataElement::EncodedBlobData>(element.data))
            return true;
    }
    return false;
}

Ref<FormData> FormData::resolveBlobReferences(BlobRegistryImpl* blobRegistryImpl)
{
    // First check if any blobs needs to be resolved, or we can take the fast path.
    if (!containsBlobElement())
        return *this;

    // Create a copy to append the result into.
    auto newFormData = FormData::create();
    newFormData->setAlwaysStream(alwaysStream());
    newFormData->setIdentifier(identifier());

    for (auto& element : m_elements) {
        switchOn(element.data,
            [&] (const Vector<uint8_t>& bytes) {
                newFormData->appendData(bytes.span());
            }, [&] (const FormDataElement::EncodedFileData& fileData) {
                newFormData->appendFileRange(fileData.filename, fileData.fileStart, fileData.fileLength, fileData.expectedFileModificationTime);
            }, [&] (const FormDataElement::EncodedBlobData& blobData) {
                appendBlobResolved(blobRegistryImpl ? blobRegistryImpl : blobRegistry().blobRegistryImpl(), newFormData.get(), blobData.url);
            }
        );
    }
    return newFormData;
}

FormDataForUpload FormData::prepareForUpload()
{
    Vector<String> generatedFiles;
    for (auto& element : m_elements) {
        auto* fileData = std::get_if<FormDataElement::EncodedFileData>(&element.data);
        if (!fileData)
            continue;
        if (FileSystem::fileTypeFollowingSymlinks(fileData->filename) != FileSystem::FileType::Directory)
            continue;
        if (fileData->fileStart || fileData->fileLength != BlobDataItem::toEndOfFile)
            continue;
        if (!fileData->fileModificationTimeMatchesExpectation())
            continue;

        auto generatedFilename = FileSystem::createTemporaryZipArchive(fileData->filename);
        if (!generatedFilename)
            continue;
        fileData->filename = generatedFilename;
        generatedFiles.append(WTFMove(generatedFilename));
    }
    
    return { *this, WTFMove(generatedFiles) };
}

FormDataForUpload::FormDataForUpload(FormData& data, Vector<String>&& temporaryZipFiles)
    : m_data(data)
    , m_temporaryZipFiles(WTFMove(temporaryZipFiles))
{
}

FormDataForUpload::~FormDataForUpload()
{
    ASSERT(isMainThread());
    for (auto& file : m_temporaryZipFiles)
        FileSystem::deleteFile(file);
}

uint64_t FormData::lengthInBytes() const
{
    if (!m_lengthInBytes) {
        uint64_t length = 0;
        for (auto& element : m_elements)
            length += element.lengthInBytes();
        m_lengthInBytes = length;
    }
    return *m_lengthInBytes;
}

RefPtr<SharedBuffer> FormData::asSharedBuffer() const
{
    for (auto& element : m_elements) {
        if (!std::holds_alternative<Vector<uint8_t>>(element.data))
            return nullptr;
    }
    return SharedBuffer::create(flatten());
}

URL FormData::asBlobURL() const
{
    if (m_elements.size() != 1)
        return { };

    if (auto* blobData = std::get_if<FormDataElement::EncodedBlobData>(&m_elements.first().data))
        return blobData->url;
    return { };
}

bool FormDataElement::EncodedFileData::fileModificationTimeMatchesExpectation() const
{
    if (!expectedFileModificationTime)
        return true;

    auto fileModificationTime = FileSystem::fileModificationTime(filename);
    if (!fileModificationTime)
        return false;

    if (fileModificationTime->secondsSinceEpoch().secondsAs<time_t>() != expectedFileModificationTime->secondsSinceEpoch().secondsAs<time_t>())
        return false;

    return true;
}

} // namespace WebCore
