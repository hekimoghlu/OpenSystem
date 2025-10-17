/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 7, 2022.
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

// Â© 2016 and later: Unicode, Inc. and others.
// License & terms of use: http://www.unicode.org/copyright.html
/**
 *******************************************************************************
 * Copyright (C) 2001-2012, International Business Machines Corporation and    *
 * others. All Rights Reserved.                                                *
 *******************************************************************************
 */

#include "unicode/utypes.h"

#if !UCONFIG_NO_SERVICE

#include "servnotf.h"
#ifdef NOTIFIER_DEBUG
#include <stdio.h>
#endif

U_NAMESPACE_BEGIN

EventListener::~EventListener() {}
UOBJECT_DEFINE_RTTI_IMPLEMENTATION(EventListener)

static UMutex notifyLock;

ICUNotifier::ICUNotifier() 
: listeners(nullptr) 
{
}

ICUNotifier::~ICUNotifier() {
    {
        Mutex lmx(&notifyLock);
        delete listeners;
        listeners = nullptr;
    }
}


void 
ICUNotifier::addListener(const EventListener* l, UErrorCode& status) 
{
    if (U_SUCCESS(status)) {
        if (l == nullptr) {
            status = U_ILLEGAL_ARGUMENT_ERROR;
            return;
        }

        if (acceptsListener(*l)) {
            Mutex lmx(&notifyLock);
            if (listeners == nullptr) {
                LocalPointer<UVector> lpListeners(new UVector(5, status), status);
                if (U_FAILURE(status)) {
                    return;
                }
                listeners = lpListeners.orphan();
            } else {
                for (int i = 0, e = listeners->size(); i < e; ++i) {
                    const EventListener* el = static_cast<const EventListener*>(listeners->elementAt(i));
                    if (l == el) {
                        return;
                    }
                }
            }

            listeners->addElement((void*)l, status); // cast away const
        }
#ifdef NOTIFIER_DEBUG
        else {
            fprintf(stderr, "Listener invalid for this notifier.");
            exit(1);
        }
#endif
    }
}

void 
ICUNotifier::removeListener(const EventListener *l, UErrorCode& status) 
{
    if (U_SUCCESS(status)) {
        if (l == nullptr) {
            status = U_ILLEGAL_ARGUMENT_ERROR;
            return;
        }

        {
            Mutex lmx(&notifyLock);
            if (listeners != nullptr) {
                // identity equality check
                for (int i = 0, e = listeners->size(); i < e; ++i) {
                    const EventListener* el = static_cast<const EventListener*>(listeners->elementAt(i));
                    if (l == el) {
                        listeners->removeElementAt(i);
                        if (listeners->size() == 0) {
                            delete listeners;
                            listeners = nullptr;
                        }
                        return;
                    }
                }
            }
        }
    }
}

void 
ICUNotifier::notifyChanged() 
{
    Mutex lmx(&notifyLock);
    if (listeners != nullptr) {
        for (int i = 0, e = listeners->size(); i < e; ++i) {
            EventListener* el = static_cast<EventListener*>(listeners->elementAt(i));
            notifyListener(*el);
        }
    }
}

U_NAMESPACE_END

/* UCONFIG_NO_SERVICE */
#endif

