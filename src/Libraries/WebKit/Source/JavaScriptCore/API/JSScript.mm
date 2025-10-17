/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 7, 2023.
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
#import "JSBase.h"
#import "JSScriptInternal.h"

#import "APICast.h"
#import "BytecodeCacheError.h"
#import "CachedTypes.h"
#import "CodeCache.h"
#import "Completion.h"
#import "Identifier.h"
#import "IntegrityInlines.h"
#import "JSContextInternal.h"
#import "JSScriptSourceProvider.h"
#import "JSSourceCode.h"
#import "JSValuePrivate.h"
#import "JSVirtualMachineInternal.h"
#import "Symbol.h"
#import <sys/stat.h>
#import <wtf/FileSystem.h>
#import <wtf/SHA1.h>
#import <wtf/SafeStrerror.h>
#import <wtf/Scope.h>
#import <wtf/StdLibExtras.h>
#import <wtf/WeakObjCPtr.h>
#import <wtf/spi/darwin/DataVaultSPI.h>
#import <wtf/text/MakeString.h>

#if JSC_OBJC_API_ENABLED

@implementation JSScript {
    WeakObjCPtr<JSVirtualMachine> m_virtualMachine;
    JSScriptType m_type;
    FileSystem::MappedFileData m_mappedSource;
    String m_source;
    RetainPtr<NSURL> m_sourceURL;
    RetainPtr<NSURL> m_cachePath;
    RefPtr<JSC::CachedBytecode> m_cachedBytecode;
}

static JSScript *createError(NSString *message, NSError** error)
{
    if (error)
        *error = [NSError errorWithDomain:@"JSScriptErrorDomain" code:1 userInfo:@{ @"message": message }];
    return nil;
}

static bool validateBytecodeCachePath(NSURL* cachePath, NSError** error)
{
    if (!cachePath)
        return true;

    URL cachePathURL([cachePath absoluteURL]);
    if (!cachePathURL.protocolIsFile()) {
        createError([NSString stringWithFormat:@"Cache path `%@` is not a local file", static_cast<NSURL *>(cachePathURL)], error);
        return false;
    }

    String systemPath = cachePathURL.fileSystemPath();

    if (auto fileType = FileSystem::fileType(systemPath)) {
        if (*fileType != FileSystem::FileType::Regular) {
            createError([NSString stringWithFormat:@"Cache path `%@` already exists and is not a file", static_cast<NSString *>(systemPath)], error);
            return false;
        }
    }

    String directory = FileSystem::parentPath(systemPath);
    if (directory.isNull()) {
        createError([NSString stringWithFormat:@"Cache path `%@` does not contain in a valid directory", static_cast<NSString *>(systemPath)], error);
        return false;
    }

    if (FileSystem::fileType(directory) != FileSystem::FileType::Directory) {
        createError([NSString stringWithFormat:@"Cache directory `%@` is not a directory or does not exist", static_cast<NSString *>(directory)], error);
        return false;
    }

#if USE(APPLE_INTERNAL_SDK)
    if (rootless_check_datavault_flag(FileSystem::fileSystemRepresentation(directory).data(), nullptr)) {
        createError([NSString stringWithFormat:@"Cache directory `%@` is not a data vault", static_cast<NSString *>(directory)], error);
        return false;
    }
#endif

    return true;
}

+ (instancetype)scriptOfType:(JSScriptType)type withSource:(NSString *)source andSourceURL:(NSURL *)sourceURL andBytecodeCache:(NSURL *)cachePath inVirtualMachine:(JSVirtualMachine *)vm error:(out NSError **)error
{
    if (!validateBytecodeCachePath(cachePath, error))
        return nil;

    auto result = adoptNS([[JSScript alloc] init]);
    result->m_virtualMachine = vm;
    result->m_type = type;
    result->m_source = source;
    result->m_sourceURL = sourceURL;
    result->m_cachePath = cachePath;
    [result readCache];
    return result.autorelease();
}

+ (instancetype)scriptOfType:(JSScriptType)type memoryMappedFromASCIIFile:(NSURL *)filePath withSourceURL:(NSURL *)sourceURL andBytecodeCache:(NSURL *)cachePath inVirtualMachine:(JSVirtualMachine *)vm error:(out NSError **)error
{
    if (!validateBytecodeCachePath(cachePath, error))
        return nil;

    URL filePathURL([filePath absoluteURL]);
    if (!filePathURL.protocolIsFile())
        return createError([NSString stringWithFormat:@"File path %@ is not a local file", static_cast<NSURL *>(filePathURL)], error);

    bool success = false;
    String systemPath = filePathURL.fileSystemPath();
    FileSystem::MappedFileData fileData(systemPath, FileSystem::MappedFileMode::Shared, success);
    if (!success)
        return createError([NSString stringWithFormat:@"File at path %@ could not be mapped.", static_cast<NSString *>(systemPath)], error);

    if (!charactersAreAllASCII(fileData.span()))
        return createError([NSString stringWithFormat:@"Not all characters in file at %@ are ASCII.", static_cast<NSString *>(systemPath)], error);

    auto result = adoptNS([[JSScript alloc] init]);
    result->m_virtualMachine = vm;
    result->m_type = type;
    result->m_source = String(StringImpl::createWithoutCopying(fileData.span()));
    result->m_mappedSource = WTFMove(fileData);
    result->m_sourceURL = sourceURL;
    result->m_cachePath = cachePath;
    [result readCache];
    return result.autorelease();
}

- (void)readCache
{
    if (!m_cachePath)
        return;

    String cacheFilename = [m_cachePath path];

    auto fd = FileSystem::openAndLockFile(cacheFilename, FileSystem::FileOpenMode::Read, {FileSystem::FileLockMode::Exclusive, FileSystem::FileLockMode::Nonblocking});
    if (!FileSystem::isHandleValid(fd))
        return;
    auto closeFD = makeScopeExit([&] {
        FileSystem::unlockAndCloseFile(fd);
    });

    bool success;
    FileSystem::MappedFileData mappedFile(fd, FileSystem::MappedFileMode::Private, success);
    if (!success)
        return;

    auto fileData = mappedFile.span();

    // Ensure we at least have a SHA1::Digest to read.
    if (fileData.size() < sizeof(SHA1::Digest)) {
        FileSystem::deleteFile(cacheFilename);
        return;
    }

    unsigned fileDataSize = fileData.size() - sizeof(SHA1::Digest);

    SHA1::Digest computedHash;
    SHA1 sha1;
    sha1.addBytes(fileData.first(fileDataSize));
    sha1.computeHash(computedHash);

    SHA1::Digest fileHash;
    auto hashSpan = fileData.subspan(fileDataSize, sizeof(SHA1::Digest));
    memcpySpan(std::span { fileHash }, hashSpan);

    if (computedHash != fileHash) {
        FileSystem::deleteFile(cacheFilename);
        return;
    }

    Ref<JSC::CachedBytecode> cachedBytecode = JSC::CachedBytecode::create(WTFMove(mappedFile));

    JSC::VM& vm = *toJS([m_virtualMachine JSContextGroupRef]);
    JSC::SourceCode sourceCode = [self sourceCode];
    JSC::SourceCodeKey key = m_type == kJSScriptTypeProgram ? sourceCodeKeyForSerializedProgram(vm, sourceCode) : sourceCodeKeyForSerializedModule(vm, sourceCode);
    if (isCachedBytecodeStillValid(vm, cachedBytecode.copyRef(), key, m_type == kJSScriptTypeProgram ? JSC::SourceCodeType::ProgramType : JSC::SourceCodeType::ModuleType))
        m_cachedBytecode = WTFMove(cachedBytecode);
    else
        FileSystem::truncateFile(fd, 0);
}

- (BOOL)cacheBytecodeWithError:(NSError **)error
{
    String errorString { };
    [self writeCache:errorString];
    if (!errorString.isNull()) {
        createError(errorString, error);
        return NO;
    }

    return YES;
}

- (BOOL)isUsingBytecodeCache
{
    return !!m_cachedBytecode->size();
}

- (NSURL *)sourceURL
{
    return m_sourceURL.get();
}

- (JSScriptType)type
{
    return m_type;
}

@end

@implementation JSScript(Internal)

- (instancetype)init
{
    self = [super init];
    if (!self)
        return nil;

    self->m_cachedBytecode = JSC::CachedBytecode::create();

    return self;
}

- (unsigned)hash
{
    return m_source.hash();
}

- (const String&)source
{
    return m_source;
}

- (RefPtr<JSC::CachedBytecode>)cachedBytecode
{
    return m_cachedBytecode;
}

- (JSC::SourceCode)sourceCode
{
    JSC::VM& vm = *toJS([m_virtualMachine JSContextGroupRef]);
    JSC::JSLockHolder locker(vm);

    TextPosition startPosition { };
    String filename = String { [[self sourceURL] absoluteString] };
    URL url = URL({ }, filename);
    auto type = m_type == kJSScriptTypeModule ? JSC::SourceProviderSourceType::Module : JSC::SourceProviderSourceType::Program;
    JSC::SourceOrigin origin(url);
    Ref<JSScriptSourceProvider> sourceProvider = JSScriptSourceProvider::create(self, origin, WTFMove(filename), String(), JSC::SourceTaintedOrigin::Untainted, startPosition, type);
    JSC::SourceCode sourceCode(WTFMove(sourceProvider), startPosition.m_line.oneBasedInt(), startPosition.m_column.oneBasedInt());
    return sourceCode;
}

- (JSC::JSSourceCode*)jsSourceCode
{
    JSC::VM& vm = *toJS([m_virtualMachine JSContextGroupRef]);
    JSC::JSLockHolder locker(vm);
    JSC::JSSourceCode* jsSourceCode = JSC::JSSourceCode::create(vm, [self sourceCode]);
    return jsSourceCode;
}

- (BOOL)writeCache:(String&)error
{
    if (self.isUsingBytecodeCache) {
        error = "Cache for JSScript is already non-empty. Can not override it."_s;
        return NO;
    }

    if (!m_cachePath) {
        error = "No cache path was provided during construction of this JSScript."_s;
        return NO;
    }

    // We want to do the write as a transaction (i.e. we guarantee that it's all
    // or nothing). So, we'll write to a temp file first, and rename the temp
    // file to the cache file only after we've finished writing the whole thing.

    NSString *cachePathString = [m_cachePath path];
    const char* cacheFileName = cachePathString.UTF8String;
    const char* tempFileName = [cachePathString stringByAppendingString:@".tmp"].UTF8String;
    int fd = open(cacheFileName, O_CREAT | O_WRONLY | O_EXLOCK | O_NONBLOCK, 0600);
    if (fd == -1) {
        error = makeString("Could not open or lock the bytecode cache file. It's likely another VM or process is already using it. Error: "_s, safeStrerror(errno).span());
        return NO;
    }

    auto closeFD = makeScopeExit([&] {
        close(fd);
    });

    int tempFD = open(tempFileName, O_CREAT | O_RDWR | O_EXLOCK | O_NONBLOCK, 0600);
    if (tempFD == -1) {
        error = makeString("Could not open or lock the bytecode cache temp file. Error: "_s, safeStrerror(errno).span());
        return NO;
    }

    auto closeTempFD = makeScopeExit([&] {
        close(tempFD);
    });

    JSC::BytecodeCacheError cacheError;
    JSC::SourceCode sourceCode = [self sourceCode];
    JSC::VM& vm = *toJS([m_virtualMachine JSContextGroupRef]);
    switch (m_type) {
    case kJSScriptTypeModule:
        m_cachedBytecode = JSC::generateModuleBytecode(vm, sourceCode, tempFD, cacheError);
        break;
    case kJSScriptTypeProgram:
        m_cachedBytecode = JSC::generateProgramBytecode(vm, sourceCode, tempFD, cacheError);
        break;
    }

    if (cacheError.isValid()) {
        m_cachedBytecode = JSC::CachedBytecode::create();
        FileSystem::truncateFile(fd, 0);
        error = makeString("Unable to generate bytecode for this JSScript because: "_s, cacheError.message());
        return NO;
    }

    SHA1::Digest computedHash;
    SHA1 sha1;
    sha1.addBytes(m_cachedBytecode->span());
    sha1.computeHash(computedHash);
    FileSystem::writeToFile(tempFD, computedHash);

    fsync(tempFD);
    rename(tempFileName, cacheFileName);
    return YES;
}

@end

#endif
