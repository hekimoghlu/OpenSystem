/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 13, 2024.
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
//
// adornment - generic attached-storage facility
//
// Adornments are dynamic objects (subclasses of class Adornment) that can
// be "attached" ("hung off") any object derived from Adornable. Any number
// of Adornments can be attached to one object using different unique keys
// (of type void *).
//
// Adornments can be used by a single caller to remember data "with" an Adornable
// object. Multiple, cooperating callers can share an Adornment as long as they
// agree on the Key.
//
// Memory management: All Adornments must be dynamically allocated, and will be
// deleted when their Adornable dies. Once attached, their memory is owned by the
// Adornable (NOT the caller). Do not get fancy with an Adornment's memory;
// trying to share one Adornment instance between Adornables or slots is bad.
// If you need shared storage, use a RefPointer attachment.
//
// Your Adornment's destructor will be called when its Adornable dies, or when
// its slot is replaced (whichever happens sooner). So you CAN get notification
// of an object's death by attaching an Adornment with a unique key and putting
// code in its destructor.
//
// It is fairly popular for a subclass of Adornable to rename its getAdornment and
// adornment methods as operator [], but we won't make that decision for you
// at this level.
//
#ifndef _H_ADORNMENTS
#define _H_ADORNMENTS

#include <security_utilities/utilities.h>
#include <security_utilities/threading.h>
#include <map>


namespace Security {

class Adornable;


//
// An Adornment is data "hung" (stored with) an Adornable.
//
class Adornment {
	friend class Adornable;
public:
	typedef const void *Key;
	
	virtual ~Adornment() = 0;
	
protected:
	Adornment() { }
};


//
// An Adornable can carry Adornments, potentially a different one for each
// Key. We provide both a raw interface (dealing in Adornment subclasses),
// and an attachment form that just pretends that the Adornable has extra,
// dynamically allocated members filed under various keys.
//
class Adornable {
public:
	Adornable() : mAdornments(NULL) { }
	~Adornable();
	
	// adornment keys (slots)
	typedef Adornment::Key Key;
	
	// primitive access, raw form
	Adornment *getAdornment(Key key) const;				// NULL if not present
	void setAdornment(Key key, Adornment *ad);			// use NULL to delete
	Adornment *swapAdornment(Key key, Adornment *ad);	// rotate in/out
	
	// typed primitive access. Ad must be a unique subclass of Adornment
	template <class Ad>
	Ad *getAdornment(Key key) const
	{ return safe_cast<Ad *>(getAdornment(key)); }

	template <class Ad>
	Ad *swapAdornment(Key key, Ad *ad)
	{ return safe_cast<Ad *>(swapAdornment(key, ad)); }
	
	// inquiries for the Adornable itself
	bool empty() const				{ return !mAdornments || mAdornments->empty(); }
	unsigned int size() const		{ return mAdornments ? (unsigned int)mAdornments->size() : 0; }
	void clearAdornments();
	
public:
	// Adornment ref interface.  Will return an (optionally constructed) Adornment &.
	template <class T> T &adornment(Key key);
	template <class T, class Arg1> T &adornment(Key key, Arg1 &arg1);
	template <class T, class Arg1, class Arg2> T &adornment(Key key, Arg1 &arg1, Arg2 &arg2);
	template <class T, class Arg1, class Arg2, class Arg3> T &adornment(Key key, Arg1 &arg1, Arg2 &arg2, Arg3 &arg3);

	// attached-value interface
	template <class T> T &attachment(Key key);
	template <class T, class Arg1> T &attachment(Key key, Arg1 arg1);

private:
	Adornment *&adornmentSlot(Key key);

	template <class Type>
	struct Attachment : public Adornment {
		Attachment() { }
		template <class Arg1> Attachment(Arg1 arg) : mValue(arg) { }
		Type mValue;
	};

private:
	typedef std::map<Key, Adornment *> AdornmentMap;
	AdornmentMap *mAdornments;
};


//
// Out-of-line implementations
//
template <class T> T &
Adornable::adornment(Key key)
{
	Adornment *&slot = adornmentSlot(key);
	if (!slot)
		slot = new T;
	return dynamic_cast<T &>(*slot);
}

template <class T, class Arg1> T &
Adornable::adornment(Key key, Arg1 &arg1)
{
	Adornment *&slot = adornmentSlot(key);
	if (!slot)
		slot = new T(arg1);
	return dynamic_cast<T &>(*slot);
}

template <class T, class Arg1, class Arg2> T &
Adornable::adornment(Key key, Arg1 &arg1, Arg2 &arg2)
{
	Adornment *&slot = adornmentSlot(key);
	if (!slot)
		slot = new T(arg1, arg2);
	return dynamic_cast<T &>(*slot);
}

template <class T, class Arg1, class Arg2, class Arg3> T &
Adornable::adornment(Key key, Arg1 &arg1, Arg2 &arg2, Arg3 &arg3)
{
	Adornment *&slot = adornmentSlot(key);
	if (!slot)
		slot = new T(arg1, arg2, arg3);
	return dynamic_cast<T &>(*slot);
}

template <class T>
T &Adornable::attachment(Key key)
{
	typedef Attachment<T> Attach;
	Adornment *&slot = adornmentSlot(key);
	if (!slot)
		slot = new Attach;
	return safe_cast<Attach *>(slot)->mValue;
}

template <class T, class Arg1>
T &Adornable::attachment(Key key, Arg1 arg1)
{
	typedef Attachment<T> Attach;
	Adornment *&slot = adornmentSlot(key);
	if (!slot)
		slot = new Attach(arg1);
	return safe_cast<Attach *>(slot)->mValue;
}


}	// end namespace Security

#endif //_H_ADORNMENTS
