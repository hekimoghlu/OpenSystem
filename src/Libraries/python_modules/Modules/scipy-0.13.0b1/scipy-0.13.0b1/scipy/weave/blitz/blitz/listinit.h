/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 4, 2025.
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
 * Initialization lists provide a convenient way to set the elements
 * of an array.  For example,
 *
 * Array<int,2> A(3,3);
 * A = 1, 0, 0,
 *     0, 1, 0,
 *     0, 0, 1;
 */

#ifndef BZ_LISTINIT_H
#define BZ_LISTINIT_H

BZ_NAMESPACE(blitz)

template<typename T_numtype, typename T_iterator>
class ListInitializer {

public:
    ListInitializer(T_iterator iter)
        : iter_(iter)
    {
    }

    ListInitializer<T_numtype, T_iterator> operator,(T_numtype x)
    {
        *iter_ = x;
        return ListInitializer<T_numtype, T_iterator>(iter_ + 1);
    }

private:
    ListInitializer();

protected:
    T_iterator iter_;
};

template<typename T_array, typename T_iterator = _bz_typename T_array::T_numtype*>
class ListInitializationSwitch {

public:
    typedef _bz_typename T_array::T_numtype T_numtype;

    ListInitializationSwitch(const ListInitializationSwitch<T_array>& lis)
        : array_(lis.array_), value_(lis.value_), 
          wipeOnDestruct_(true)
    {
        lis.disable();
    }

    ListInitializationSwitch(T_array& array, T_numtype value)
        : array_(array), value_(value), wipeOnDestruct_(true)
    { }

    ~ListInitializationSwitch()
    {
        if (wipeOnDestruct_)
            array_.initialize(value_);
    }

    ListInitializer<T_numtype, T_iterator> operator,(T_numtype x)
    {
        wipeOnDestruct_ = false;
        T_iterator iter = array_.getInitializationIterator();
        *iter = value_;
        T_iterator iter2 = iter + 1;
        *iter2 = x;
        return ListInitializer<T_numtype, T_iterator>(iter2 + 1);
    }

    void disable() const
    {
        wipeOnDestruct_ = false;
    }

private:
    ListInitializationSwitch();

protected:
    T_array& array_;
    T_numtype value_;
    mutable bool wipeOnDestruct_;
};

BZ_NAMESPACE_END

#endif // BZ_LISTINIT_H

