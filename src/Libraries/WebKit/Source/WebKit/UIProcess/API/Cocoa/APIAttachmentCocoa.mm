/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 14, 2021.
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
#import "config.h"
#import "APIAttachment.h"

#import "PageClient.h"
#import "WebPageProxy.h"
#import <WebCore/MIMETypeRegistry.h>
#import <WebCore/SharedBuffer.h>

#if PLATFORM(IOS_FAMILY)
#import <MobileCoreServices/MobileCoreServices.h>
#else
#import <CoreServices/CoreServices.h>
#endif

namespace API {

static WTF::String mimeTypeInferredFromFileExtension(const API::Attachment& attachment)
{
    if (NSString *fileExtension = [(NSString *)attachment.fileName() pathExtension])
        return WebCore::MIMETypeRegistry::mimeTypeForExtension(WTF::String(fileExtension));

    return { };
}

static BOOL isDeclaredOrDynamicTypeIdentifier(NSString *type)
{
ALLOW_DEPRECATED_DECLARATIONS_BEGIN
    return UTTypeIsDeclared((__bridge CFStringRef)type) || UTTypeIsDynamic((__bridge CFStringRef)type);
ALLOW_DEPRECATED_DECLARATIONS_END
}

void Attachment::setFileWrapper(NSFileWrapper *fileWrapper)
{
    Locker locker { m_fileWrapperLock };

    m_fileWrapper = fileWrapper;
}

void Attachment::doWithFileWrapper(Function<void(NSFileWrapper *)>&& function) const
{
    Locker locker { m_fileWrapperLock };

    function(m_fileWrapper.get());
}

WTF::String Attachment::mimeType() const
{
    NSString *contentType = m_contentType.isEmpty() ? mimeTypeInferredFromFileExtension(*this) : m_contentType;
    if (!contentType.length)
        return nullString();
    if (!isDeclaredOrDynamicTypeIdentifier(contentType))
        return contentType;

ALLOW_DEPRECATED_DECLARATIONS_BEGIN
    return adoptCF(UTTypeCopyPreferredTagWithClass((__bridge CFStringRef)contentType, kUTTagClassMIMEType)).get();
ALLOW_DEPRECATED_DECLARATIONS_END
}

WTF::String Attachment::utiType() const
{
    NSString *contentType = m_contentType.isEmpty() ? mimeTypeInferredFromFileExtension(*this) : m_contentType;
    if (!contentType.length)
        return nullString();
    if (isDeclaredOrDynamicTypeIdentifier(contentType))
        return contentType;

ALLOW_DEPRECATED_DECLARATIONS_BEGIN
    return adoptCF(UTTypeCreatePreferredIdentifierForTag(kUTTagClassMIMEType, (__bridge CFStringRef)contentType, nullptr)).get();
ALLOW_DEPRECATED_DECLARATIONS_END
}

WTF::String Attachment::fileName() const
{
    Locker locker { m_fileWrapperLock };

    if ([m_fileWrapper filename].length)
        return [m_fileWrapper filename];

    return [m_fileWrapper preferredFilename];
}

void Attachment::setFileWrapperAndUpdateContentType(NSFileWrapper *fileWrapper, NSString *contentType)
{
    if (!contentType.length) {
ALLOW_DEPRECATED_DECLARATIONS_BEGIN
        if (fileWrapper.directory)
            contentType = (NSString *)kUTTypeDirectory;
        else if (fileWrapper.regularFile) {
            if (NSString *pathExtension = (fileWrapper.filename.length ? fileWrapper.filename : fileWrapper.preferredFilename).pathExtension)
                contentType = WebCore::MIMETypeRegistry::mimeTypeForExtension(WTF::String(pathExtension));
            if (!contentType.length)
                contentType = (NSString *)kUTTypeData;
        }
ALLOW_DEPRECATED_DECLARATIONS_END
    }

    setContentType(contentType);
    setFileWrapper(fileWrapper);
}

std::optional<uint64_t> Attachment::fileSizeForDisplay() const
{
    Locker locker { m_fileWrapperLock };

    if (![m_fileWrapper isRegularFile]) {
        // FIXME: We should display a size estimate for directory-type file wrappers.
        return std::nullopt;
    }

    if (auto fileSize = [[m_fileWrapper fileAttributes][NSFileSize] unsignedLongLongValue])
        return fileSize;

    return [m_fileWrapper regularFileContents].length;
}

RefPtr<WebCore::FragmentedSharedBuffer> Attachment::associatedElementData() const
{
    if (m_associatedElementType == WebCore::AttachmentAssociatedElementType::None)
        return nullptr;

    NSData *data = nil;
    {
        Locker locker { m_fileWrapperLock };

        if (![m_fileWrapper isRegularFile])
            return nullptr;

        data = [m_fileWrapper regularFileContents];
    }

    if (!data)
        return nullptr;

    return WebCore::SharedBuffer::create(data);
}

NSData *Attachment::associatedElementNSData() const
{
    Locker locker { m_fileWrapperLock };

    if (![m_fileWrapper isRegularFile])
        return nil;

    return [m_fileWrapper regularFileContents];
}

bool Attachment::isEmpty() const
{
    Locker locker { m_fileWrapperLock };

    return !m_fileWrapper;
}

RefPtr<WebCore::SharedBuffer> Attachment::createSerializedRepresentation() const
{
    NSData *serializedData = nil;
    {
        Locker locker { m_fileWrapperLock };

        if (!m_fileWrapper || !m_webPage)
            return nullptr;

        serializedData = [NSKeyedArchiver archivedDataWithRootObject:m_fileWrapper.get() requiringSecureCoding:YES error:nullptr];
    }
    if (!serializedData)
        return nullptr;

    return WebCore::SharedBuffer::create(serializedData);
}

void Attachment::updateFromSerializedRepresentation(Ref<WebCore::SharedBuffer>&& serializedRepresentation, const WTF::String& contentType)
{
    if (!m_webPage)
        return;

    RefPtr pageClient = m_webPage->pageClient();
    if (!pageClient)
        return;

    auto serializedData = serializedRepresentation->createNSData();
    if (!serializedData)
        return;

    RetainPtr fileWrapper = [NSKeyedUnarchiver unarchivedObjectOfClasses:pageClient->serializableFileWrapperClasses() fromData:serializedData.get() error:nullptr];
    if (![fileWrapper isKindOfClass:NSFileWrapper.class])
        return;

    m_isCreatedFromSerializedRepresentation = true;
    setFileWrapperAndUpdateContentType(fileWrapper.get(), contentType);
    m_webPage->updateAttachmentAttributes(*this, [] { });
}

void Attachment::cloneFileWrapperTo(Attachment& other)
{
    other.m_isCreatedFromSerializedRepresentation = m_isCreatedFromSerializedRepresentation;

    Locker locker { m_fileWrapperLock };
    other.setFileWrapper(m_fileWrapper.get());
}

bool Attachment::shouldUseFileWrapperIconForDirectory() const
{
    if (m_contentType != "public.directory"_s)
        return false;

    if (m_isCreatedFromSerializedRepresentation)
        return false;

    {
        Locker locker { m_fileWrapperLock };
        if (![m_fileWrapper isDirectory])
            return false;
    }

    return true;
}

} // namespace API
