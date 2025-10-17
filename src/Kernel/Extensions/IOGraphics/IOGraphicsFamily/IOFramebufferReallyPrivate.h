/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 5, 2024.
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
class _IOFramebufferNotifier : public IONotifier
{
    friend class IOFramebuffer;

    OSDeclareDefaultStructors(_IOFramebufferNotifier)

public:
    OSOrderedSet *                      fWhence;

    IOFramebufferNotificationHandler    fHandler;
    OSObject *                          fTarget;
    void *                              fRef;
    bool                                fEnable;
    int32_t                             fGroup;
    IOIndex                             fGroupPriority;
    IOSelect                            fEvents;
    IOSelect                            fLastEvent;

    char                                fName[64];
    uint64_t                            fStampStart;
    uint64_t                            fStampEnd;

    virtual void remove() APPLE_KEXT_OVERRIDE;
    virtual bool disable() APPLE_KEXT_OVERRIDE;
    virtual void enable( bool was ) APPLE_KEXT_OVERRIDE;

    bool init(IOFramebufferNotificationHandler handler, OSObject * target, void * ref,
              IOIndex groupPriority, IOSelect events, int32_t groupIndex);
};
