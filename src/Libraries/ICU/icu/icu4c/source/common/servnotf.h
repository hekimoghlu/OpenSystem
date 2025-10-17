/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 11, 2023.
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
 * Copyright (C) 2001-2014, International Business Machines Corporation and    *
 * others. All Rights Reserved.                                                *
 *******************************************************************************
 */
#ifndef ICUNOTIF_H
#define ICUNOTIF_H

#include "unicode/utypes.h"

#if UCONFIG_NO_SERVICE

U_NAMESPACE_BEGIN

/*
 * Allow the declaration of APIs with pointers to BreakIterator
 * even when break iteration is removed from the build.
 */
class ICUNotifier;

U_NAMESPACE_END

#else

#include "unicode/uobject.h"
#include "unicode/unistr.h"

#include "mutex.h"
#include "uvector.h"

U_NAMESPACE_BEGIN

class U_COMMON_API EventListener : public UObject {
public: 
    virtual ~EventListener();

public:
    static UClassID U_EXPORT2 getStaticClassID();

    virtual UClassID getDynamicClassID() const override;

public:
#ifdef SERVICE_DEBUG
    virtual UnicodeString& debug(UnicodeString& result) const {
      return debugClass(result);
    }

    virtual UnicodeString& debugClass(UnicodeString& result) const {
      return result.append((UnicodeString)"Key");
    }
#endif
};

/**
 * <p>Abstract implementation of a notification facility.  Clients add
 * EventListeners with addListener and remove them with removeListener.
 * Notifiers call notifyChanged when they wish to notify listeners.
 * This queues the listener list on the notification thread, which
 * eventually dequeues the list and calls notifyListener on each
 * listener in the list.</p>
 *
 * <p>Subclasses override acceptsListener and notifyListener 
 * to add type-safe notification.  AcceptsListener should return
 * true if the listener is of the appropriate type; ICUNotifier
 * itself will ensure the listener is non-null and that the
 * identical listener is not already registered with the Notifier.
 * NotifyListener should cast the listener to the appropriate 
 * type and call the appropriate method on the listener.
 */

class U_COMMON_API ICUNotifier : public UMemory  {
private: UVector* listeners;
         
public: 
    ICUNotifier();
    
    virtual ~ICUNotifier();
    
    /**
     * Add a listener to be notified when notifyChanged is called.
     * The listener must not be null. AcceptsListener must return
     * true for the listener. Attempts to concurrently
     * register the identical listener more than once will be
     * silently ignored.  
     */
    virtual void addListener(const EventListener* l, UErrorCode& status);
    
    /**
     * Stop notifying this listener.  The listener must
     * not be null. Attempts to remove a listener that is
     * not registered will be silently ignored.
     */
    virtual void removeListener(const EventListener* l, UErrorCode& status);
    
    /**
     * ICU doesn't spawn its own threads.  All listeners are notified in
     * the thread of the caller.  Misbehaved listeners can therefore
     * indefinitely block the calling thread.  Callers should beware of
     * deadlock situations.  
     */
    virtual void notifyChanged();
    
protected: 
    /**
     * Subclasses implement this to return true if the listener is
     * of the appropriate type.
     */
    virtual UBool acceptsListener(const EventListener& l) const = 0;
    
    /**
     * Subclasses implement this to notify the listener.
     */
    virtual void notifyListener(EventListener& l) const = 0;
};

U_NAMESPACE_END

/* UCONFIG_NO_SERVICE */
#endif

/* ICUNOTIF_H */
#endif
