/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 4, 2024.
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

#import <Foundation/NSObject.h>
#import <Foundation/NSMethodSignature.h>

@class NSMutableDictionary;
@class NSArray;
@class NSSet;
@class NSEnumerator;

@interface MVChatPluginManager : NSObject {
	@private
	NSMutableDictionary *_plugins;
}
+ (MVChatPluginManager *) defaultManager;

- (NSArray *) pluginSearchPaths;
- (void) findAndLoadPlugins;

- (NSSet *) plugins;
- (NSSet *) pluginsThatRespondToSelector:(SEL) selector;
- (NSSet *) pluginsOfClass:(Class) class thatRespondToSelector:(SEL) selector;

- (NSEnumerator *) pluginEnumerator;
- (NSEnumerator *) enumeratorOfPluginsThatRespondToSelector:(SEL) selector;
- (NSEnumerator *) enumeratorOfPluginsOfClass:(Class) class thatRespondToSelector:(SEL) selector;

- (unsigned int) numberOfPlugins;
- (unsigned int) numberOfPluginsThatRespondToSelector:(SEL) selector;
- (unsigned int) numberOfPluginsOfClass:(Class) class thatRespondToSelector:(SEL) selector;

- (NSArray *) makePluginsPerformInvocation:(NSInvocation *) invocation;
- (NSArray *) makePluginsPerformInvocation:(NSInvocation *) invocation stoppingOnFirstSuccessfulReturn:(BOOL) stop;
- (NSArray *) makePluginsOfClass:(Class) class performInvocation:(NSInvocation *) invocation stoppingOnFirstSuccessfulReturn:(BOOL) stop;
@end

@protocol MVChatPlugin
- (id) initWithManager:(MVChatPluginManager *) manager;
@end
