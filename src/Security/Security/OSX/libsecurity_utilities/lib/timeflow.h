/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 15, 2025.
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
// timeflow - abstract view of the flow of time
//
// We happily publish both absolute and relative times as floating-point values.
// Absolute times are off the UNIX Epoch (1/1/1970, midnight). This leaves us about
// microsecond resolution in Modern Times.
//
#ifndef _H_TIMEFLOW
#define _H_TIMEFLOW

#include <sys/time.h>
#include <limits.h>
#include <math.h>	// for MAXFLOAT


namespace Security {
namespace Time {


//
// A Time::Interval is a time difference (distance).
//
class Interval {
    friend class Absolute;
public:
    Interval() { }
    Interval(int seconds)		{ mValue = seconds; }
    Interval(double seconds)	{ mValue = seconds; }
    explicit Interval(time_t seconds) { mValue = seconds; }
    
    Interval &operator += (Interval rel)	{ mValue += rel.mValue; return *this; }
    Interval &operator -= (Interval rel)	{ mValue -= rel.mValue; return *this; }
    Interval &operator *= (double f)		{ mValue *= f; return *this; }
    Interval &operator /= (double f)		{ mValue /= f; return *this; }
    
    bool operator <  (Interval other) const	{ return mValue < other.mValue; }
    bool operator <= (Interval other) const	{ return mValue <= other.mValue; }
    bool operator >  (Interval other) const	{ return mValue > other.mValue; }
    bool operator >= (Interval other) const	{ return mValue >= other.mValue; }
    bool operator == (Interval other) const	{ return mValue == other.mValue; }
    bool operator != (Interval other) const	{ return mValue != other.mValue; }
    
    // express as (fractions of) seconds, milliseconds, or microseconds
    double seconds() const					{ return mValue; }
    double mSeconds() const					{ return mValue * 1E3; }
    double uSeconds() const					{ return mValue * 1E6; }
    
    // struct timeval is sometimes used for time intervals, but not often - so be explicit
    struct timeval timevalInterval() const;

private:
    double mValue;
};


//
// A Time::Absolute is a moment in time.
//
class Absolute {
    friend class Interval;
    friend Interval operator - (Absolute, Absolute);
    friend Absolute now();
    friend Absolute bigBang();
    friend Absolute heatDeath();
public:
    Absolute() { }						// uninitialized
    Absolute(time_t t) { mValue = t; }	// from time_t
    Absolute(const struct timeval &tv);	// from timeval
	Absolute(const struct timespec &ts); // from timespec
    
    // *crement operators
    Absolute &operator += (Interval rel)	{ mValue += rel.mValue; return *this; }
    Absolute &operator -= (Interval rel)	{ mValue -= rel.mValue; return *this; }

    // comparisons
    bool operator <  (Absolute other) const	{ return mValue < other.mValue; }
    bool operator <= (Absolute other) const	{ return mValue <= other.mValue; }
    bool operator >  (Absolute other) const	{ return mValue > other.mValue; }
    bool operator >= (Absolute other) const	{ return mValue >= other.mValue; }
    bool operator == (Absolute other) const	{ return mValue == other.mValue; }
    bool operator != (Absolute other) const	{ return mValue != other.mValue; }
    
    // express as conventional (absolute!) time measures
    operator struct timeval() const;
	operator struct timespec() const;
    operator time_t () const				{ return time_t(mValue); }

    // internal form for debugging ONLY
    double internalForm() const				{ return mValue; }
    
private:
    double mValue;
    
    Absolute(double value) : mValue(value) { }
};


//
// Time::now produces the current time
//
Absolute now();						// get "now"


//
// Time::resolution(when) gives a conservative estimate of the available resolution
// at a given time.
//
Interval resolution(Absolute at);	// estimate available resolution at given time


//
// Some useful "constants"
//
inline Absolute bigBang()				{ return -MAXFLOAT; }
inline Absolute heatDeath()				{ return +MAXFLOAT; }



//
// More inline arithmetic
//
inline Interval operator + (Interval r, Interval r2)	{ r += r2; return r; }
inline Interval operator - (Interval r, Interval r2)	{ r -= r2; return r; }
inline Interval operator * (Interval r, double f)		{ r *= f; return r; }
inline Interval operator / (Interval r, double f)		{ r /= f; return r; }

inline Absolute operator + (Absolute a, Interval r)		{ return a += r; }
inline Absolute operator + (Interval r, Absolute a)		{ return a += r; }
inline Absolute operator - (Absolute a, Interval r)		{ return a -= r; }

inline Interval operator - (Absolute t1, Absolute t0)
{ return t1.mValue - t0.mValue; }


}	// end namespace Time
}	// end namespace Security

#endif //_H_TIMEFLOW
