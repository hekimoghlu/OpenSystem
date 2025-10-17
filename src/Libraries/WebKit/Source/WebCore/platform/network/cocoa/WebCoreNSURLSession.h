/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 22, 2025.
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
#import "RangeResponseGenerator.h"
#import "SecurityOrigin.h"
#import <Foundation/NSURLSession.h>
#import <wtf/CompletionHandler.h>
#import <wtf/HashSet.h>
#import <wtf/Lock.h>
#import <wtf/OSObjectPtr.h>
#import <wtf/Ref.h>
#import <wtf/RefPtr.h>
#import <wtf/RetainPtr.h>
#import <wtf/WeakObjCPtr.h>

@class NSNetService;
@class NSOperationQueue;
@class NSURL;
@class NSURLRequest;
@class NSURLResponse;
@class WebCoreNSURLSessionDataTask;

namespace WebCore {
class CachedResourceRequest;
class NetworkLoadMetrics;
class PlatformMediaResource;
class PlatformMediaResourceLoader;
class ResourceError;
class ResourceRequest;
class ResourceResponse;
class FragmentedSharedBuffer;
class SharedBufferDataView;
class WebCoreNSURLSessionDataTaskClient;
enum class ShouldContinuePolicyCheck : bool;
}

namespace WTF {
class GuaranteedSerialFunctionDispatcher;
class WorkQueue;
}

enum class WebCoreNSURLSessionCORSAccessCheckResults : uint8_t {
    Unknown,
    Pass,
    Fail,
};

NS_ASSUME_NONNULL_BEGIN

// Created on the main thread; used on targetQueue.
WEBCORE_EXPORT @interface WebCoreNSURLSession : NSObject {
@private
    RefPtr<WebCore::PlatformMediaResourceLoader> _loader;
    RefPtr<GuaranteedSerialFunctionDispatcher> _targetDispatcher;
    WeakObjCPtr<id<NSURLSessionDelegate>> _delegate;
    RetainPtr<NSOperationQueue> _queue;
    RetainPtr<NSString> _sessionDescription;
    Lock _dataTasksLock;
    UncheckedKeyHashSet<RetainPtr<WebCoreNSURLSessionDataTask>> _dataTasks WTF_GUARDED_BY_LOCK(_dataTasksLock);
    UncheckedKeyHashSet<RefPtr<WebCore::SecurityOrigin>> _origins WTF_GUARDED_BY_LOCK(_dataTasksLock);
    BOOL _invalidated; // Access on multiple thread, must be accessed via ObjC property.
    NSUInteger _nextTaskIdentifier;
    RefPtr<WTF::WorkQueue> _internalQueue;
    WebCoreNSURLSessionCORSAccessCheckResults _corsResults;
    RefPtr<WebCore::RangeResponseGenerator> _rangeResponseGenerator; // Only created/accessed on _targetQueue
}
- (id)initWithResourceLoader:(WebCore::PlatformMediaResourceLoader&)loader delegate:(id<NSURLSessionTaskDelegate>)delegate delegateQueue:(NSOperationQueue*)queue;
@property (readonly, retain) NSOperationQueue *delegateQueue;
@property (nullable, readonly) id <NSURLSessionDelegate> delegate;
@property (readonly, copy) NSURLSessionConfiguration *configuration;
@property (copy) NSString *sessionDescription;
@property (readonly) BOOL didPassCORSAccessChecks;
@property (assign, atomic) WebCoreNSURLSessionCORSAccessCheckResults corsResults;
@property (assign, atomic) BOOL invalidated;
- (void)finishTasksAndInvalidate;
- (void)invalidateAndCancel;
- (BOOL)isCrossOrigin:(const WebCore::SecurityOrigin&)origin;

- (void)resetWithCompletionHandler:(void (^)(void))completionHandler;
- (void)flushWithCompletionHandler:(void (^)(void))completionHandler;
- (void)getTasksWithCompletionHandler:(void (^)(NSArray<NSURLSessionDataTask *> *dataTasks, NSArray<NSURLSessionUploadTask *> *uploadTasks, NSArray<NSURLSessionDownloadTask *> *downloadTasks))completionHandler;
- (void)getAllTasksWithCompletionHandler:(void (^)(NSArray<__kindof NSURLSessionTask *> *tasks))completionHandler;

- (NSURLSessionDataTask *)dataTaskWithRequest:(NSURLRequest *)request;
- (NSURLSessionDataTask *)dataTaskWithURL:(NSURL *)url;
- (NSURLSessionUploadTask *)uploadTaskWithRequest:(NSURLRequest *)request fromFile:(NSURL *)fileURL;
- (NSURLSessionUploadTask *)uploadTaskWithRequest:(NSURLRequest *)request fromData:(NSData *)bodyData;
- (NSURLSessionUploadTask *)uploadTaskWithStreamedRequest:(NSURLRequest *)request;
- (NSURLSessionDownloadTask *)downloadTaskWithRequest:(NSURLRequest *)request;
- (NSURLSessionDownloadTask *)downloadTaskWithURL:(NSURL *)url;
- (NSURLSessionDownloadTask *)downloadTaskWithResumeData:(NSData *)resumeData;
- (NSURLSessionStreamTask *)streamTaskWithHostName:(NSString *)hostname port:(NSInteger)port;
- (NSURLSessionStreamTask *)streamTaskWithNetService:(NSNetService *)service;
@end

@interface WebCoreNSURLSession (NSURLSessionAsynchronousConvenience)
- (NSURLSessionDataTask *)dataTaskWithRequest:(NSURLRequest *)request completionHandler:(void (^)(NSData * data, NSURLResponse * response, NSError * error))completionHandler;
- (NSURLSessionDataTask *)dataTaskWithURL:(NSURL *)url completionHandler:(void (^)(NSData * data, NSURLResponse * response, NSError * error))completionHandler;
- (NSURLSessionUploadTask *)uploadTaskWithRequest:(NSURLRequest *)request fromFile:(NSURL *)fileURL completionHandler:(void (^)(NSData * data, NSURLResponse * response, NSError * error))completionHandler;
- (NSURLSessionUploadTask *)uploadTaskWithRequest:(NSURLRequest *)request fromData:(nullable NSData *)bodyData completionHandler:(void (^)(NSData * data, NSURLResponse * response, NSError * error))completionHandler;
- (NSURLSessionDownloadTask *)downloadTaskWithRequest:(NSURLRequest *)request completionHandler:(void (^)(NSURL * location, NSURLResponse * response, NSError * error))completionHandler;
- (NSURLSessionDownloadTask *)downloadTaskWithURL:(NSURL *)url completionHandler:(void (^)(NSURL * location, NSURLResponse * response, NSError * error))completionHandler;
- (NSURLSessionDownloadTask *)downloadTaskWithResumeData:(NSData *)resumeData completionHandler:(void (^)(NSURL * location, NSURLResponse * response, NSError * error))completionHandler;
@end

@interface WebCoreNSURLSession (WebKitAwesomeness)
- (void)sendH2Ping:(NSURL *)url pongHandler:(void (^)(NSError * _Nullable error, NSTimeInterval interval))pongHandler;
@end

// Created on com.apple.avfoundation.customurl.nsurlsession
@interface WebCoreNSURLSessionDataTask : NSObject {
    WeakObjCPtr<WebCoreNSURLSession> _session; // Accesssed from operation queue, main and loader thread. Must be accessed through Obj-C property.
    RefPtr<GuaranteedSerialFunctionDispatcher> _targetDispatcher;
    RefPtr<WebCore::PlatformMediaResource> _resource; // Accesssed from loader thread.
    RetainPtr<NSURLResponse> _response; // Set on operation queue.
    NSUInteger _taskIdentifier;
    RetainPtr<NSURLRequest> _originalRequest; // Set on construction, never modified.
    RetainPtr<NSURLRequest> _currentRequest; // Set on construction, never modified.
    int64_t _countOfBytesReceived;
    int64_t _countOfBytesSent;
    int64_t _countOfBytesExpectedToSend;
    int64_t _countOfBytesExpectedToReceive;
    std::atomic<NSURLSessionTaskState> _state;
    RetainPtr<NSError> _error; // Unused, always nil.
    RetainPtr<NSString> _taskDescription; // Only set / read on the user's thread.
    float _priority;
    uint32_t _resumeSessionID; // Accesssed from loader thread.
}

@property NSUInteger taskIdentifier;
@property (nullable, readonly, copy) NSURLRequest *originalRequest;
@property (nullable, readonly, copy) NSURLRequest *currentRequest;
@property (nullable, readonly, copy) NSURLResponse *response;
@property (assign, atomic) int64_t countOfBytesReceived;
@property int64_t countOfBytesSent;
@property int64_t countOfBytesExpectedToSend;
@property int64_t countOfBytesExpectedToReceive;
@property (readonly) NSURLSessionTaskState state;
@property (nullable, readonly, copy) NSError *error;
@property (nullable, copy) NSString *taskDescription;
@property float priority;
- (void)cancel;
- (void)suspend;
- (void)resume;
@end

@interface WebCoreNSURLSessionDataTask (WebKitInternal)
- (void)resource:(nullable WebCore::PlatformMediaResource*)resource sentBytes:(unsigned long long)bytesSent totalBytesToBeSent:(unsigned long long)totalBytesToBeSent;
- (void)resource:(nullable WebCore::PlatformMediaResource*)resource receivedResponse:(const WebCore::ResourceResponse&)response completionHandler:(CompletionHandler<void(WebCore::ShouldContinuePolicyCheck)>&&)completionHandler;
- (BOOL)resource:(nullable WebCore::PlatformMediaResource*)resource shouldCacheResponse:(const WebCore::ResourceResponse&)response;
- (void)resource:(nullable WebCore::PlatformMediaResource*)resource receivedData:(RetainPtr<NSData>&&)data;
- (void)resource:(nullable WebCore::PlatformMediaResource*)resource receivedRedirect:(const WebCore::ResourceResponse&)response request:(WebCore::ResourceRequest&&)request completionHandler:(CompletionHandler<void(WebCore::ResourceRequest&&)>&&)completionHandler;
- (void)resource:(nullable WebCore::PlatformMediaResource*)resource accessControlCheckFailedWithError:(const WebCore::ResourceError&)error;
- (void)resource:(nullable WebCore::PlatformMediaResource*)resource loadFailedWithError:(const WebCore::ResourceError&)error;
- (void)resourceFinished:(nullable WebCore::PlatformMediaResource*)resource metrics:(const WebCore::NetworkLoadMetrics&)metrics;
@end

NS_ASSUME_NONNULL_END
