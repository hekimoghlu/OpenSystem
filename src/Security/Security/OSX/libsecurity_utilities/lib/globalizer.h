/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 17, 2021.
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
/*
 * globalizer - multiscope globalization services
 */
#ifndef _H_GLOBALIZER
#define _H_GLOBALIZER

#include <security_utilities/threading.h>
#include <memory>
#include <set>
#include <dispatch/dispatch.h>
#include <libkern/OSAtomic.h>
#include <os/lock.h>

namespace Security {


//
// GlobalNexus is the common superclass of all globality scopes.
// A Nexus is an *access point* to the *single* object of a given
// type in the Nexus's particular scope.
//
class GlobalNexus {
public:
    class Error : public std::exception {
    public:
        virtual ~Error() _NOEXCEPT;
        const char * const message;
        Error(const char *m) : message(m) { }
        const char *what() const _NOEXCEPT { return message; }
    };
};


class ModuleNexusCommon : public GlobalNexus {
private:
    void do_create(void *(*make)());

protected:
    void *create(void *(*make)());
    void lock() {os_unfair_lock_lock(&access);}
    void unlock() {os_unfair_lock_unlock(&access);}

protected:
    // all of these will be statically initialized to zero
	void *pointer;
    dispatch_once_t once;
    os_unfair_lock access;
};

template <class Type>
class ModuleNexus : public ModuleNexusCommon {
public:
    Type &operator () () 
    {
        lock();
        
        try
        {
            if (pointer == NULL)
            {
                pointer = create(make);
            }
            
            unlock();
        }
        catch (...)
        {
            unlock();
            throw;
        }
        
		return *reinterpret_cast<Type *>(pointer);
    }
	
	// does the object DEFINITELY exist already?
	bool exists() const
	{
        bool result;
        lock();
        result = pointer != NULL;
        unlock();
        return result;
	}
    
	// destroy the object (if any) and start over - not really thread-safe
    void reset()
    {
        lock();
        if (pointer != NULL)
        {
            delete reinterpret_cast<Type *>(pointer);
            pointer = NULL;
            once = 0;
        }
        unlock();
    }
    
private:
    static void *make() { return new Type; }
};

template <class Type>
class CleanModuleNexus : public ModuleNexus<Type> {
public:
    ~CleanModuleNexus()
    {
        secinfo("nexus", "ModuleNexus %p destroyed object 0x%x",
			this, ModuleNexus<Type>::pointer);
        delete reinterpret_cast<Type *>(ModuleNexus<Type>::pointer);
    }
};

typedef std::set<void*> RetentionSet;

//
// A thread-scope nexus is tied to a particular native thread AND
// a particular nexus object. Its scope is all code in any one thread
// that access that particular Nexus object. Any number of Nexus objects
// can exist, and each implements a different scope for each thread.
// NOTE: ThreadNexus is dynamically constructed. If you want static,
// zero-initialization ThreadNexi, put them inside a ModuleNexus.
//
template <class Type>
class ThreadNexus : public GlobalNexus {
public:
    ThreadNexus() : mSlot(true) { }

    Type &operator () ()
    {
        // no thread contention here!
        if (Type *p = mSlot)
            return *p;
        mSlot = new Type;
        return *mSlot;
    }

private:
    PerThreadPointer<Type> mSlot;
};


} // end namespace Security

#endif //_H_GLOBALIZER
