/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 20, 2023.
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
// transactions - generic transaction frame support
//
#ifndef _H_TRANSACTIONS
#define _H_TRANSACTIONS

#include <security_utilities/utilities.h>
#include <security_utilities/debugging.h>


namespace Security {


//
// Implementation base class. Do not use directly.
//
class TransactionBase {
public:
	// what happens if this object gets destroyed?
	enum Outcome {
		successful,						// succeeds as set
		canceled,						// canceled (rolled back)
		conditional						// succeeds normally, canceled on exception
	};
	
public:
    virtual ~TransactionBase();
	
	void outcome(Outcome oc)	{ mOutcome = oc; }
	Outcome outcome() const		{ return mOutcome; }

protected:
	TransactionBase(Outcome outcome) : mOutcome(outcome) { }

	Outcome finalOutcome() const;

private:
	Outcome mOutcome;					// current outcome setting
};


//
// A ManagedTransaction will call methods begin() and end() on the Carrier object
// it belongs to, and manage the "outcome" state and semantics automatically.
// You would usually subclass this, though the class is complete in itself if you
// need nothing else out of your transaction objects.
//
template <class Carrier>
class ManagedTransaction : public TransactionBase {
public:
	ManagedTransaction(Carrier &carrier, Outcome outcome = conditional)
		: TransactionBase(outcome), mCarrier(carrier)
	{
		carrier.begin();
	}
	
	~ManagedTransaction()
	{
		switch (finalOutcome()) {
		case successful:
			this->commitAction();
			break;
		case canceled:
			this->cancelAction();
			break;
		default:
			assert(false);
			break;
		}
	}

protected:
	virtual void commitAction()		{ mCarrier.end(); }
	virtual void cancelAction()		{ mCarrier.cancel(); }

	Carrier &mCarrier;
};


}	// end namespace Security


#endif //_H_TRANSACTIONS
