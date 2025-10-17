/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 12, 2024.
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

#include <stdlib.h>
#include <notify.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <darwintest.h>
#include <signal.h>

#define KEY "com.apple.notify.test.notify_register_signal"

#define HANDLED 1
#define UNHANDLED 0

// Notification token
int n_token;

// Global to show if notification was handled
int handled = 0;

void post_notification(char * msg, int should_handle) {
    int i, loops = 5000;   //5000*10 = 50,000 = 50ms

    // Post notification
    handled = UNHANDLED;
    T_LOG("%s", msg);
    notify_post(KEY);

    for(i=0; i < loops; i++) {
        if(handled == HANDLED)
            break;
        else
            usleep(10);
    }

    if(should_handle)
        T_EXPECT_EQ(handled, (should_handle ? HANDLED : UNHANDLED),
                "Signal based notification handled as expected.");
}

void setup() {

    // Register with notification system
    dispatch_queue_t dq = dispatch_queue_create(NULL, DISPATCH_QUEUE_SERIAL);
    int rv = notify_register_dispatch(KEY, &n_token, dq, ^(int token){
        handled = HANDLED;

        qos_class_t block_qos = qos_class_self();
        if (block_qos != QOS_CLASS_DEFAULT){
            T_FAIL("Block is running at QoS %#x instead of DEF", block_qos);
        } else {
        }
    });
    if (rv)
        T_FAIL("Unable to notify_register_dispatch");
}

void cleanup() {
    notify_cancel(n_token); /* Releases the queue - block must finish executing */
}

T_DECL(notify_qos,
        "Test that work for notification runs at qos of target queue",
       T_META("owner", "Core Darwin Daemons & Tools"),
       T_META("as_root", "false"))
{
    setup();

    post_notification("Ensure notifications are being handled properly.", 1);

    cleanup();
}
