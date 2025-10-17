/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 23, 2022.
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
#ifndef HIDSessionFilter_h
#define HIDSessionFilter_h

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

// Forward declarations
@class HIDEventService;
@class HIDEvent;
@class HIDSession;
@class HIDConnection;

/*!
 * @protocol HIDServiceFilter
 *
 * @abstract
 * A protocol for creating HID session filters.
 *
 * @discussion
 * HID session filters match to the HID session and are able to filter calls that the
 * session receives.
 */
@protocol HIDSessionFilter <NSObject>

/*!
 * @method initWithSession
 *
 * @abstract
 * Create a HIDSessionFilter for the corresponding session.
 *
 * @result
 * A HIDSessionFilter instance on success, nil on failure.
 */
- (nullable instancetype)initWithSession:(HIDSession *)session;

/*!
 * @method propertyForKey
 *
 * @abstract
 * Obtain a property from the session filter.
 *
 * @param key
 * The property key being queried.
 *
 * @result
 * The property on success, nil on failure.
 */
- (nullable id)propertyForKey:(NSString *)key;

/*!
 * @method setProperty
 *
 * @abstract
 * Set a property on the session filter.
 *
 * @param value
 * The value of the property to set.
 *
 * @param key
 * The property key to set.
 *
 * @result
 * true on success, false on failure.
 */
- (BOOL)setProperty:(nullable id)value
             forKey:(NSString *)key;

/*!
 * @method filterEvent
 *
 * @abstract
 * Filter an event for the provided service.
 *
 * @discussion
 * The filterEvent method provides the session filter with a stream of events
 * from every service. The session filter may observe, modify, or drop the event
 * if it chooses. If the filter is only observing the events, it should return
 * the event unmodified.
 *
 * @param event
 * The event to filter.
 *
 * @param service
 * The service associated with the event.
 *
 * @result
 * A filtered event, or nil if the event should be dropped.
 */
- (nullable HIDEvent *)filterEvent:(HIDEvent *)event
                        forService:(HIDEventService *)service;

/*!
 * @method activate
 *
 * @abstract
 * Activate the session filter.
 *
 * @discussion
 * A HIDSessionFilter is created in an inactive state. The session will
 * be activated after it has been initialized, and the setDispatchQueue and
 * setCancelHandler methods are invoked.
 */
- (void)activate;

@optional

/*!
 * @method cancel
 *
 * @abstract
 * Cancel the session filter.
 *
 * @discussion
 * This function will called prior to the release of the filter and provides the
 * opportunity to cleanup any necessary state before the session is destroyed.
 */
- (void)cancel;

/*!
 * @method serviceNotification
 *
 * @abstract
 * Method invoked when a service is added or removed from the HID event system.
 *
 * @discussion
 * The filter should release any strong references to the service when the
 * service is removed.
 *
 * @param service
 * The service that is being enumerated/terminated.
 *
 * @param added
 * True if the service is enumerated, false if it is terminating.
 */
- (void)serviceNotification:(HIDEventService *)service added:(BOOL)added;

/*!
 * @method setDispatchQueue
 *
 * @abstract
 * Provides the session filter with a dispatch queue to be used for
 * synchronization and handling asynchronous tasks.
 *
 * @discussion
 * Please note: The dispatch queue provided here should be used only for filter
 * related work. Kernel calls, or calls that make take some time should be made
 * on a separate queue, so as not to hold up the whole HID event system.
 *
 * @param queue
 * The dispatch queue to be used by the session filter.
 */
- (void)setDispatchQueue:(dispatch_queue_t)queue;

/*!
 * @method filterEvent
 *
 * @abstract
 * Filter an event from the provided service to the provided connection.
 *
 * @discussion
 * The filterEvent method provides the session filter with a stream of events
 * from every service. The session filter may observe, modify, or drop the event
 * if it chooses. This variant of filterEvent can filter events as they branch to be sent
 * to specific connections. Filtering out an event will not filter copies of the event sent
 * to other connections.
 *
 * @param event
 * The event to filter.
 *
 * @param connection
 * The connection that would receive the event.
 *
 * @param service
 * The service associated with the event.
 *
 * @result
 * A filtered event, or nil if the event should be dropped.
 */
- (nullable HIDEvent *)filterEvent:(HIDEvent *)event
                      toConnection:(HIDConnection *)connection
                       fromService:(HIDEventService *)service;

@end

NS_ASSUME_NONNULL_END

#endif /* HIDSessionFilter_h */
