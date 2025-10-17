/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 13, 2025.
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
#import <WebKit/_WKAttachment.h>

#import "APIAttachment.h"
#import "WKErrorPrivate.h"
#import "_WKAttachmentInternal.h"
#import <WebCore/MIMETypeRegistry.h>
#import <WebCore/SharedBuffer.h>
#import <WebCore/WebCoreObjCExtras.h>
#import <wtf/BlockPtr.h>
#import <wtf/CompletionHandler.h>

#if PLATFORM(IOS_FAMILY)
#import <MobileCoreServices/MobileCoreServices.h>
#endif

static const NSInteger UnspecifiedAttachmentErrorCode = 1;
static const NSInteger InvalidAttachmentErrorCode = 2;

@implementation _WKAttachmentDisplayOptions : NSObject
@end

@implementation _WKAttachmentInfo {
    RefPtr<const API::Attachment> _attachment;
    RetainPtr<NSString> _mimeType;
    RetainPtr<NSString> _utiType;
    RetainPtr<NSString> _filePath;
}

- (instancetype)initWithAttachment:(const API::Attachment&)attachment
{
    if (!(self = [super init]))
        return nil;

    _attachment = &attachment;
    _filePath = attachment.filePath();
    _mimeType = attachment.mimeType();
    _utiType = attachment.utiType();
    return self;
}

- (NSData *)data
{
    NSData *result = nil;
    _attachment->doWithFileWrapper([&](NSFileWrapper *fileWrapper) {
        // FIXME: Handle attachments backed by NSFileWrappers that represent directories.
        result = fileWrapper.isRegularFile ? fileWrapper.regularFileContents : nil;
    });
    return result;
}

- (NSString *)name
{
    NSString *result = nil;
    _attachment->doWithFileWrapper([&](NSFileWrapper *fileWrapper) {
        result = fileWrapper.filename.length ? fileWrapper.filename : fileWrapper.preferredFilename;
    });
    return result;
}

- (NSString *)filePath
{
    return _filePath.get();
}

- (NSFileWrapper *)fileWrapper
{
    // FIXME: This API is potentially unsafe for WebKit clients, since the file wrapper that's
    // returned could be simultaneously accessed from a background thread, due to QuickLook
    // thumbnailing. This should be replaced with a method that instead takes a callback, and
    // invokes with callback with a file wrapper in a way that guarantees thread safety.
    NSFileWrapper *result = nil;
    _attachment->doWithFileWrapper([&](NSFileWrapper *fileWrapper) {
        result = fileWrapper;
    });
    return result;
}

- (NSString *)contentType
{
    if ([_mimeType length])
        return _mimeType.get();

    return _utiType.get();
}

- (BOOL)shouldPreserveFidelity
{
    return _attachment->associatedElementType() == WebCore::AttachmentAssociatedElementType::Source;
}

@end

@implementation _WKAttachment

- (void)dealloc
{
    if (WebCoreObjCScheduleDeallocateOnMainRunLoop(_WKAttachment.class, self))
        return;

    _attachment->~Attachment();

    [super dealloc];
}

- (API::Object&)_apiObject
{
    return *_attachment;
}

- (_WKAttachmentInfo *)info
{
    if (!_attachment->isValid())
        return nil;

    return adoptNS([[_WKAttachmentInfo alloc] initWithAttachment:*_attachment]).autorelease();
}

- (void)requestInfo:(void(^)(_WKAttachmentInfo *, NSError *))completionHandler
{
    completionHandler(self.info, nil);
}

- (void)setFileWrapper:(NSFileWrapper *)fileWrapper contentType:(NSString *)contentType completion:(void (^)(NSError *))completionHandler
{
    if (!_attachment->isValid()) {
        if (completionHandler)
            completionHandler([NSError errorWithDomain:WKErrorDomain code:InvalidAttachmentErrorCode userInfo:nil]);
        return;
    }

    // This file path member is only populated when the attachment is generated upon dropping files. When data is specified via NSFileWrapper
    // from the SPI client, the corresponding file path of the data is unknown, if it even exists at all.
    _attachment->setFilePath({ });
    _attachment->setFileWrapperAndUpdateContentType(fileWrapper, contentType);
    _attachment->updateAttributes([capturedBlock = makeBlockPtr(completionHandler)] {
        if (capturedBlock)
            capturedBlock(nil);
    });
}

- (void)setData:(NSData *)data newContentType:(NSString *)newContentType
{
    auto fileWrapper = adoptNS([[NSFileWrapper alloc] initRegularFileWithContents:data]);
    [self setFileWrapper:fileWrapper.get() contentType:newContentType completion:nil];
}

- (void)setData:(NSData *)data newContentType:(NSString *)newContentType newFilename:(NSString *)newFilename completion:(void(^)(NSError *))completionHandler
{
    auto fileWrapper = adoptNS([[NSFileWrapper alloc] initRegularFileWithContents:data]);
    if (newFilename)
        [fileWrapper setPreferredFilename:newFilename];
    [self setFileWrapper:fileWrapper.get() contentType:newContentType completion:completionHandler];
}

- (NSString *)uniqueIdentifier
{
    return _attachment->identifier();
}

- (NSString *)description
{
    return [NSString stringWithFormat:@"<%@ %p id='%@'>", [self class], self, self.uniqueIdentifier];
}

- (BOOL)isConnected
{
    return _attachment->insertionState() == API::Attachment::InsertionState::Inserted;
}

@end
