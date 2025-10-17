/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 20, 2023.
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
#if !__has_feature(objc_arc)
#error This file requires ARC. Add the "-fobjc-arc" compiler flag for this file.
#endif

#import "config.h"
#import "WKWebExtensionControllerConfigurationInternal.h"

#import "APIPageConfiguration.h"
#import "WKWebViewConfigurationPrivate.h"
#import "WKWebsiteDataStoreInternal.h"
#import "WebExtensionControllerConfiguration.h"

#if ENABLE(WK_WEB_EXTENSIONS)
static NSString * const persistentCodingKey = @"persistent";
static NSString * const temporaryCodingKey = @"temporary";
static NSString * const temporaryDirectoryCodingKey = @"temporaryDirectory";
static NSString * const identifierCodingKey = @"identifier";
static NSString * const webViewConfigurationCodingKey = @"webViewConfiguration";
static NSString * const defaultWebsiteDataStoreCodingKey = @"defaultWebsiteDataStore";
#endif

@implementation WKWebExtensionControllerConfiguration

+ (BOOL)supportsSecureCoding
{
    return YES;
}

#if ENABLE(WK_WEB_EXTENSIONS)

WK_OBJECT_DEALLOC_IMPL_ON_MAIN_THREAD(WKWebExtensionControllerConfiguration, WebExtensionControllerConfiguration, _webExtensionControllerConfiguration);

+ (instancetype)defaultConfiguration
{
    return WebKit::WebExtensionControllerConfiguration::createDefault()->wrapper();
}

+ (instancetype)nonPersistentConfiguration
{
    return WebKit::WebExtensionControllerConfiguration::createNonPersistent()->wrapper();
}

+ (instancetype)configurationWithIdentifier:(NSUUID *)identifier
{
    NSParameterAssert([identifier isKindOfClass:NSUUID.class]);

    auto uuid = WTF::UUID::fromNSUUID(identifier);
    RELEASE_ASSERT(uuid);

    return WebKit::WebExtensionControllerConfiguration::create(*uuid)->wrapper();
}

+ (instancetype)_temporaryConfiguration
{
    return WebKit::WebExtensionControllerConfiguration::createTemporary()->wrapper();
}

- (void)encodeWithCoder:(NSCoder *)coder
{
    NSParameterAssert([coder isKindOfClass:NSCoder.class]);

    [coder encodeObject:self.identifier forKey:identifierCodingKey];
    [coder encodeBool:self.persistent forKey:persistentCodingKey];
    [coder encodeObject:self.webViewConfiguration forKey:webViewConfigurationCodingKey];
    [coder encodeObject:self.defaultWebsiteDataStore forKey:defaultWebsiteDataStoreCodingKey];

    if (!self._temporary)
        return;

    [coder encodeBool:YES forKey:temporaryCodingKey];
    [coder encodeObject:self._storageDirectoryPath forKey:temporaryDirectoryCodingKey];
}

- (instancetype)initWithCoder:(NSCoder *)coder
{
    NSParameterAssert([coder isKindOfClass:NSCoder.class]);

    if (!(self = [super init]))
        return nil;

    using IsPersistent = WebKit::WebExtensionControllerConfiguration::IsPersistent;
    using TemporaryTag = WebKit::WebExtensionControllerConfiguration::TemporaryTag;

    if ([coder containsValueForKey:temporaryCodingKey]) {
        RELEASE_ASSERT([coder decodeBoolForKey:temporaryCodingKey]);

        NSString *temporaryDirectory = [coder decodeObjectOfClass:NSString.class forKey:temporaryDirectoryCodingKey];
        API::Object::constructInWrapper<WebKit::WebExtensionControllerConfiguration>(self, TemporaryTag::Temporary, temporaryDirectory);

        // Remake the directories if needed, since they might have been cleaned up since this was last used.
        FileSystem::makeAllDirectories(temporaryDirectory);

        self.webViewConfiguration = [coder decodeObjectOfClass:WKWebViewConfiguration.class forKey:webViewConfigurationCodingKey];
        self.defaultWebsiteDataStore = [coder decodeObjectOfClass:WKWebsiteDataStore.class forKey:defaultWebsiteDataStoreCodingKey];

        return self;
    }

    NSUUID *identifier = [coder decodeObjectOfClass:NSUUID.class forKey:identifierCodingKey];
    BOOL persistent = [coder decodeBoolForKey:persistentCodingKey];

    if (auto uuid = WTF::UUID::fromNSUUID(identifier))
        API::Object::constructInWrapper<WebKit::WebExtensionControllerConfiguration>(self, *uuid);
    else
        API::Object::constructInWrapper<WebKit::WebExtensionControllerConfiguration>(self, persistent ? IsPersistent::Yes : IsPersistent::No);

    self.webViewConfiguration = [coder decodeObjectOfClass:WKWebViewConfiguration.class forKey:webViewConfigurationCodingKey];
    self.defaultWebsiteDataStore = [coder decodeObjectOfClass:WKWebsiteDataStore.class forKey:defaultWebsiteDataStoreCodingKey];

    return self;
}

- (id)copyWithZone:(NSZone *)zone
{
    return self._protectedWebExtensionControllerConfiguration->copy()->wrapper();
}

- (BOOL)isEqual:(id)object
{
    if (self == object)
        return YES;

    auto *other = dynamic_objc_cast<WKWebExtensionControllerConfiguration>(object);
    if (!other)
        return NO;

    return *_webExtensionControllerConfiguration == *other->_webExtensionControllerConfiguration;
}

- (NSString *)debugDescription
{
    return [NSString stringWithFormat:@"<%@: %p; persistent = %@; temporary = %@; identifier = %@>", NSStringFromClass(self.class), self, self.persistent ? @"YES" : @"NO", self._temporary ? @"YES" : @"NO", self.identifier];
}

- (NSUUID *)identifier
{
    if (auto identifier = self._protectedWebExtensionControllerConfiguration->identifier())
        return identifier.value();
    return nil;
}

- (BOOL)isPersistent
{
    return _webExtensionControllerConfiguration->storageIsPersistent();
}

- (WKWebViewConfiguration *)webViewConfiguration
{
    return self._protectedWebExtensionControllerConfiguration->webViewConfiguration();
}

- (void)setWebViewConfiguration:(WKWebViewConfiguration *)configuration
{
    self._protectedWebExtensionControllerConfiguration->setWebViewConfiguration(configuration);
}

- (WKWebsiteDataStore *)defaultWebsiteDataStore
{
    return wrapper(self._protectedWebExtensionControllerConfiguration->protectedDefaultWebsiteDataStore().get());
}

- (void)setDefaultWebsiteDataStore:(WKWebsiteDataStore *)dataStore
{
    RefPtr websiteDataStore = dataStore ? dataStore->_websiteDataStore.get() : nullptr;
    self._protectedWebExtensionControllerConfiguration->setDefaultWebsiteDataStore(websiteDataStore.get());
}

- (BOOL)_isTemporary
{
    return _webExtensionControllerConfiguration->storageIsTemporary();
}

- (NSString *)_storageDirectoryPath
{
    if (auto& directory = _webExtensionControllerConfiguration->storageDirectory(); !directory.isEmpty())
        return directory;
    return nil;
}

- (void)_setStorageDirectoryPath:(NSString *)path
{
    _webExtensionControllerConfiguration->setStorageDirectory(path);
}

#pragma mark WKObject protocol implementation

- (API::Object&)_apiObject
{
    return *_webExtensionControllerConfiguration;
}

- (WebKit::WebExtensionControllerConfiguration&)_webExtensionControllerConfiguration
{
    return *_webExtensionControllerConfiguration;
}

- (Ref<WebKit::WebExtensionControllerConfiguration>)_protectedWebExtensionControllerConfiguration
{
    return *_webExtensionControllerConfiguration;
}

#else // ENABLE(WK_WEB_EXTENSIONS)

+ (instancetype)defaultConfiguration
{
    return nil;
}

+ (instancetype)nonPersistentConfiguration
{
    return nil;
}

+ (instancetype)configurationWithIdentifier:(NSUUID *)identifier
{
    return nil;
}

+ (instancetype)_temporaryConfiguration
{
    return nil;
}

- (void)encodeWithCoder:(NSCoder *)coder
{
}

- (instancetype)initWithCoder:(NSCoder *)coder
{
    return nil;
}

- (id)copyWithZone:(NSZone *)zone
{
    return nil;
}

- (NSUUID *)identifier
{
    return nil;
}

- (BOOL)isPersistent
{
    return NO;
}

- (WKWebViewConfiguration *)webViewConfiguration
{
    return nil;
}

- (void)setWebViewConfiguration:(WKWebViewConfiguration *)webViewConfiguration
{
}

- (WKWebsiteDataStore *)defaultWebsiteDataStore
{
    return nil;
}

- (void)setDefaultWebsiteDataStore:(WKWebsiteDataStore *)dataStore
{
}

- (BOOL)_isTemporary
{
    return NO;
}

- (NSString *)_storageDirectoryPath
{
    return nil;
}

- (void)_setStorageDirectoryPath:(NSString *)path
{
}

#endif // ENABLE(WK_WEB_EXTENSIONS)

@end
