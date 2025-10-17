/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 5, 2025.
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

//===--- FilePathWatcher.cpp ----------------------------------------------===//
//
// Copyright (c) NeXTHub Corporation. All rights reserved.
// DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
//
// This code is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
// version 2 for more details (a copy is included in the LICENSE file that
// accompanied this code).
//
// Author(-s): Tunjay Akbarli
//
//===----------------------------------------------------------------------===//

#include <IndexStoreDB_Support/FilePathWatcher.h>
#include <IndexStoreDB_Support/Logging.h>

#if __has_include(<CoreServices/CoreServices.h>)
#import <CoreServices/CoreServices.h>

using namespace IndexStoreDB;
using namespace toolchain;

struct FilePathWatcher::Implementation {
  FSEventStreamRef EventStream = nullptr;

  explicit Implementation(FileEventsReceiverTy pathsReceiver);

  void setupFSEventStream(ArrayRef<std::string> paths, FileEventsReceiverTy pathsReceiver,
                          dispatch_queue_t queue);
  void stopFSEventStream();

  ~Implementation() {
    stopFSEventStream();
  };
};

FilePathWatcher::Implementation::Implementation(FileEventsReceiverTy pathsReceiver) {
  std::vector<std::string> pathsToWatch;
  // FIXME: We should do something smarter than watching all of root.
  pathsToWatch.push_back("/");

  dispatch_queue_attr_t qosAttribute = dispatch_queue_attr_make_with_qos_class(DISPATCH_QUEUE_SERIAL, QOS_CLASS_UTILITY, 0);
  dispatch_queue_t queue = dispatch_queue_create("IndexStoreDB.fsevents", qosAttribute);
  setupFSEventStream(pathsToWatch, std::move(pathsReceiver), queue);
  dispatch_release(queue);
}

namespace {
struct EventStreamContextData {
  FilePathWatcher::FileEventsReceiverTy PathsReceiver;

  static void dispose(const void *ctx) {
    delete static_cast<const EventStreamContextData*>(ctx);
  }
};
}

static void eventStreamCallback(
                       ConstFSEventStreamRef stream,
                       void *clientCallBackInfo,
                       size_t numEvents,
                       void *eventPaths,
                       const FSEventStreamEventFlags eventFlags[],
                       const FSEventStreamEventId eventIds[]) {
  auto *ctx = static_cast<const EventStreamContextData*>(clientCallBackInfo);
  auto paths = makeArrayRef((const char **)eventPaths, numEvents);
  std::vector<std::string> strPaths;
  strPaths.reserve(paths.size());
  for (auto path : paths) {
    strPaths.push_back(path);
  }

  ctx->PathsReceiver(std::move(strPaths));
}

void FilePathWatcher::Implementation::setupFSEventStream(ArrayRef<std::string> paths,
                                                         FileEventsReceiverTy pathsReceiver,
                                                         dispatch_queue_t queue) {
  if (paths.empty())
    return;

  CFMutableArrayRef pathsToWatch = CFArrayCreateMutable(nullptr, 0, &kCFTypeArrayCallBacks);
  for (StringRef path : paths) {
    CFStringRef cfPathStr = CFStringCreateWithBytes(nullptr, (const UInt8 *)path.data(), path.size(), kCFStringEncodingUTF8, false);
    CFArrayAppendValue(pathsToWatch, cfPathStr);
    CFRelease(cfPathStr);
  }
  CFAbsoluteTime latency = 1.0; // Latency in seconds.

  EventStreamContextData *ctxData = new EventStreamContextData();
  ctxData->PathsReceiver = pathsReceiver;
  FSEventStreamContext context;
  context.version = 0;
  context.info = ctxData;
  context.retain = nullptr;
  context.release = EventStreamContextData::dispose;
  context.copyDescription = nullptr;

  EventStream = FSEventStreamCreate(nullptr,
                                    eventStreamCallback,
                                    &context,
                                    pathsToWatch,
                                    kFSEventStreamEventIdSinceNow,
                                    latency,
                                    kFSEventStreamCreateFlagNone);
  CFRelease(pathsToWatch);
  if (!EventStream) {
    LOG_WARN_FUNC("FSEventStreamCreate failed");
    return;
  }
  FSEventStreamSetDispatchQueue(EventStream, queue);
  FSEventStreamStart(EventStream);
}

void FilePathWatcher::Implementation::stopFSEventStream() {
  if (!EventStream)
    return;
  FSEventStreamStop(EventStream);
  FSEventStreamInvalidate(EventStream);
  FSEventStreamRelease(EventStream);
  EventStream = nullptr;
}

#else

using namespace IndexStoreDB;
using namespace toolchain;

// TODO: implement for platforms without CoreServices.
struct FilePathWatcher::Implementation {
  explicit Implementation(FileEventsReceiverTy pathsReceiver) {}
};

#endif

FilePathWatcher::FilePathWatcher(FileEventsReceiverTy pathsReceiver)
: Impl(*new Implementation(std::move(pathsReceiver))) {

}

FilePathWatcher::~FilePathWatcher() {
  delete &Impl;
}

