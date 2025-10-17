/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 22, 2023.
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
#pragma once

#include "Algorithm.h"
#include "BInline.h"
#include <climits>

namespace bmalloc {

constexpr size_t bitsArrayLength(size_t numBits) { return (numBits + 31) / 32; }

class BitsWordView {
public:
    typedef BitsWordView ViewType;
    
    BitsWordView() { }
    
    BitsWordView(const uint32_t* array, size_t numBits)
        : m_words(array)
        , m_numBits(numBits)
    {
    }
    
    size_t numBits() const
    {
        return m_numBits;
    }
    
    uint32_t word(size_t index) const
    {
        RELEASE_BASSERT(index < bitsArrayLength(numBits()));
        return m_words[index];
    }
    
private:
    const uint32_t* m_words { nullptr };
    size_t m_numBits { 0 };
};

template<size_t passedNumBits>
class BitsWordOwner {
public:
    typedef BitsWordView ViewType;
    
    BitsWordOwner()
    {
        clearAll();
    }
    
    BitsWordOwner(const BitsWordOwner& other)
    {
        *this = other;
    }
    
    BitsWordView view() const { return BitsWordView(m_words, numBits()); }
    
    BitsWordOwner& operator=(const BitsWordOwner& other)
    {
        memcpy(m_words, other.m_words, arrayLength() * sizeof(uint32_t));
        return *this;
    }
    
    void setAll()
    {
        memset(m_words, 255, arrayLength() * sizeof(uint32_t));
    }
    
    void clearAll()
    {
        memset(m_words, 0, arrayLength() * sizeof(uint32_t));
    }
    
    void set(const BitsWordOwner& other)
    {
        memcpy(m_words, other.m_words, arrayLength() * sizeof(uint32_t));
    }
    
    size_t numBits() const
    {
        return passedNumBits;
    }
    
    size_t arrayLength() const
    {
        return bitsArrayLength(numBits());
    }
    
    uint32_t word(size_t index) const
    {
        RELEASE_BASSERT(index < arrayLength());
        return m_words[index];
    }
    
    uint32_t& word(size_t index)
    {
        RELEASE_BASSERT(index < arrayLength());
        return m_words[index];
    }
    
    const uint32_t* words() const { return m_words; }
    uint32_t* words() { return m_words; }

private:
    uint32_t m_words[bitsArrayLength(passedNumBits)];
};

template<typename Left, typename Right>
class BitsAndWords {
public:
    typedef BitsAndWords ViewType;
    
    BitsAndWords(const Left& left, const Right& right)
        : m_left(left)
        , m_right(right)
    {
        RELEASE_BASSERT(m_left.numBits() == m_right.numBits());
    }
    
    BitsAndWords view() const { return *this; }
    
    size_t numBits() const
    {
        return m_left.numBits();
    }
    
    uint32_t word(size_t index) const
    {
        return m_left.word(index) & m_right.word(index);
    }
    
private:
    Left m_left;
    Right m_right;
};
    
template<typename Left, typename Right>
class BitsOrWords {
public:
    typedef BitsOrWords ViewType;
    
    BitsOrWords(const Left& left, const Right& right)
        : m_left(left)
        , m_right(right)
    {
        RELEASE_BASSERT(m_left.numBits() == m_right.numBits());
    }
    
    BitsOrWords view() const { return *this; }
    
    size_t numBits() const
    {
        return m_left.numBits();
    }
    
    uint32_t word(size_t index) const
    {
        return m_left.word(index) | m_right.word(index);
    }
    
private:
    Left m_left;
    Right m_right;
};
    
template<typename View>
class BitsNotWords {
public:
    typedef BitsNotWords ViewType;
    
    BitsNotWords(const View& view)
        : m_view(view)
    {
    }
    
    BitsNotWords view() const { return *this; }
    
    size_t numBits() const
    {
        return m_view.numBits();
    }
    
    uint32_t word(size_t index) const
    {
        return ~m_view.word(index);
    }
    
private:
    View m_view;
};

template<size_t> class Bits;

template<typename Words>
class BitsImpl {
public:
    BitsImpl()
        : m_words()
    {
    }
    
    BitsImpl(const Words& words)
        : m_words(words)
    {
    }
    
    BitsImpl(Words&& words)
        : m_words(std::move(words))
    {
    }

    size_t numBits() const { return m_words.numBits(); }
    size_t size() const { return numBits(); }
    
    size_t arrayLength() const { return bitsArrayLength(numBits()); }
    
    template<typename Other>
    bool operator==(const Other& other) const
    {
        if (numBits() != other.numBits())
            return false;
        for (size_t i = arrayLength(); i--;) {
            if (m_words.word(i) != other.m_words.word(i))
                return false;
        }
        return true;
    }
    
    template<typename Other>
    bool operator!=(const Other& other) const
    {
        return !(*this == other);
    }
    
    bool at(size_t index) const
    {
        return atImpl(index);
    }
    
    bool operator[](size_t index) const
    {
        return atImpl(index);
    }
    
    bool isEmpty() const
    {
        for (size_t index = arrayLength(); index--;) {
            if (m_words.word(index))
                return false;
        }
        return true;
    }
    
    template<typename OtherWords>
    BitsImpl<BitsAndWords<typename Words::ViewType, typename OtherWords::ViewType>> operator&(const BitsImpl<OtherWords>& other) const
    {
        return BitsImpl<BitsAndWords<typename Words::ViewType, typename OtherWords::ViewType>>(BitsAndWords<typename Words::ViewType, typename OtherWords::ViewType>(wordView(), other.wordView()));
    }
    
    template<typename OtherWords>
    BitsImpl<BitsOrWords<typename Words::ViewType, typename OtherWords::ViewType>> operator|(const BitsImpl<OtherWords>& other) const
    {
        return BitsImpl<BitsOrWords<typename Words::ViewType, typename OtherWords::ViewType>>(BitsOrWords<typename Words::ViewType, typename OtherWords::ViewType>(wordView(), other.wordView()));
    }
    
    BitsImpl<BitsNotWords<typename Words::ViewType>> operator~() const
    {
        return BitsImpl<BitsNotWords<typename Words::ViewType>>(BitsNotWords<typename Words::ViewType>(wordView()));
    }
    
    template<typename Func>
    BINLINE void forEachSetBit(const Func& func) const
    {
        size_t n = arrayLength();
        for (size_t i = 0; i < n; ++i) {
            uint32_t word = m_words.word(i);
            size_t j = i * 32;
            while (word) {
                if (word & 1)
                    func(j);
                word >>= 1;
                j++;
            }
        }
    }
    
    template<typename Func>
    BINLINE void forEachClearBit(const Func& func) const
    {
        (~*this).forEachSetBit(func);
    }
    
    template<typename Func>
    void forEachBit(bool value, const Func& func) const
    {
        if (value)
            forEachSetBit(func);
        else
            forEachClearBit(func);
    }
    
    // Starts looking for bits at the index you pass. If that index contains the value you want,
    // then it will return that index. Returns numBits when we get to the end. For example, you
    // can write a loop to iterate over all set bits like this:
    //
    // for (size_t i = bits.findBit(0, true); i < bits.numBits(); i = bits.findBit(i + 1, true))
    //     ...
    BINLINE size_t findBit(size_t startIndex, bool value) const
    {
        // If value is true, this produces 0. If value is false, this produces UINT_MAX. It's
        // written this way so that it performs well regardless of whether value is a constant.
        uint32_t skipValue = -(static_cast<uint32_t>(value) ^ 1);
        
        size_t numWords = bitsArrayLength(m_words.numBits());
        
        size_t wordIndex = startIndex / 32;
        size_t startIndexInWord = startIndex - wordIndex * 32;
        
        while (wordIndex < numWords) {
            uint32_t word = m_words.word(wordIndex);
            if (word != skipValue) {
                size_t index = startIndexInWord;
                if (findBitInWord(word, index, 32, value))
                    return wordIndex * 32 + index;
            }
            
            wordIndex++;
            startIndexInWord = 0;
        }
        
        return numBits();
    }
    
    BINLINE size_t findSetBit(size_t index) const
    {
        return findBit(index, true);
    }
    
    BINLINE size_t findClearBit(size_t index) const
    {
        return findBit(index, false);
    }
    
    typename Words::ViewType wordView() const { return m_words.view(); }
    
private:
    // You'd think that we could remove this friend if we used protected, but you'd be wrong,
    // because templates.
    template<size_t> friend class Bits;
    
    bool atImpl(size_t index) const
    {
        RELEASE_BASSERT(index < numBits());
        return !!(m_words.word(index >> 5) & (1 << (index & 31)));
    }
    
    Words m_words;
};

template<size_t passedNumBits>
class Bits : public BitsImpl<BitsWordOwner<passedNumBits>> {
public:
    Bits() { }
    
    Bits(const Bits&) = default;
    Bits& operator=(const Bits&) = default;
    
    template<typename OtherWords>
    Bits(const BitsImpl<OtherWords>& other)
    {
        *this = other;
    }
    
    template<typename OtherWords>
    Bits& operator=(const BitsImpl<OtherWords>& other)
    {
        if (this->numBits() != other.numBits())
            resize(other.numBits());
        
        for (unsigned i = this->arrayLength(); i--;)
            this->m_words.word(i) = other.m_words.word(i);
        return *this;
    }
    
    void resize(size_t numBits)
    {
        this->m_words.resize(numBits);
    }
    
    void setAll()
    {
        this->m_words.setAll();
    }
    
    void clearAll()
    {
        this->m_words.clearAll();
    }
    
    // Returns true if the contents of this bitvector changed.
    template<typename OtherWords>
    bool setAndCheck(const BitsImpl<OtherWords>& other)
    {
        bool changed = false;
        RELEASE_BASSERT(this->numBits() == other.numBits());
        for (unsigned i = this->arrayLength(); i--;) {
            changed |= this->m_words.word(i) != other.m_words.word(i);
            this->m_words.word(i) = other.m_words.word(i);
        }
        return changed;
    }
    
    template<typename OtherWords>
    Bits& operator|=(const BitsImpl<OtherWords>& other)
    {
        RELEASE_BASSERT(this->numBits() == other.numBits());
        for (unsigned i = this->arrayLength(); i--;)
            this->m_words.word(i) |= other.m_words.word(i);
        return *this;
    }
    
    template<typename OtherWords>
    Bits& operator&=(const BitsImpl<OtherWords>& other)
    {
        RELEASE_BASSERT(this->numBits() == other.numBits());
        for (unsigned i = this->arrayLength(); i--;)
            this->m_words.word(i) &= other.m_words.word(i);
        return *this;
    }
    
    bool at(size_t index) const
    {
        return this->atImpl(index);
    }
    
    bool operator[](size_t index) const
    {
        return this->atImpl(index);
    }
    
    class BitReference {
    public:
        BitReference() { }
        
        BitReference(uint32_t* word, uint32_t mask)
            : m_word(word)
            , m_mask(mask)
        {
        }
        
        explicit operator bool() const
        {
            return !!(*m_word & m_mask);
        }
        
        BitReference& operator=(bool value)
        {
            if (value)
                *m_word |= m_mask;
            else
                *m_word &= ~m_mask;
            return *this;
        }
        
    private:
        uint32_t* m_word { nullptr };
        uint32_t m_mask { 0 };
    };
    
    BitReference at(size_t index)
    {
        RELEASE_BASSERT(index < this->numBits());
        return BitReference(&this->m_words.word(index >> 5), 1 << (index & 31));
    }
    
    BitReference operator[](size_t index)
    {
        return at(index);
    }
};

} // namespace bmalloc

