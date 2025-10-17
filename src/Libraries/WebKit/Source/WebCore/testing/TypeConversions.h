/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 31, 2022.
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

#include "Node.h"
#include <variant>
#include <wtf/RefCounted.h>
#include <wtf/Vector.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class TypeConversions : public RefCounted<TypeConversions> {
public:
    static Ref<TypeConversions> create() { return adoptRef(*new TypeConversions()); }

    struct OtherDictionary {
        int longValue;
        String stringValue;
    };

    struct Dictionary {
        int longValue;
        String stringValue;
        String treatNullAsEmptyStringValue;
        Vector<String> sequenceValue;
        std::variant<RefPtr<Node>, Vector<String>, OtherDictionary> unionValue;
        int clampLongValue;
        int enforceRangeLongValue;
    };

    int8_t testByte() { return m_byte; }
    void setTestByte(int8_t value) { m_byte = value; }
    int8_t testEnforceRangeByte() { return m_byte; }
    void setTestEnforceRangeByte(int8_t value) { m_byte = value; }
    int8_t testClampByte() { return m_byte; }
    void setTestClampByte(int8_t value) { m_byte = value; }
    uint8_t testOctet() { return m_octet; }
    void setTestOctet(uint8_t value) { m_octet = value; }
    uint8_t testEnforceRangeOctet() { return m_octet; }
    void setTestEnforceRangeOctet(uint8_t value) { m_octet = value; }
    uint8_t testClampOctet() { return m_octet; }
    void setTestClampOctet(uint8_t value) { m_octet = value; }

    int16_t testShort() { return m_short; }
    void setTestShort(int16_t value) { m_short = value; }
    int16_t testEnforceRangeShort() { return m_short; }
    void setTestEnforceRangeShort(int16_t value) { m_short = value; }
    int16_t testClampShort() { return m_short; }
    void setTestClampShort(int16_t value) { m_short = value; }
    uint16_t testUnsignedShort() { return m_unsignedShort; }
    void setTestUnsignedShort(uint16_t value) { m_unsignedShort = value; }
    uint16_t testEnforceRangeUnsignedShort() { return m_unsignedShort; }
    void setTestEnforceRangeUnsignedShort(uint16_t value) { m_unsignedShort = value; }
    uint16_t testClampUnsignedShort() { return m_unsignedShort; }
    void setTestClampUnsignedShort(uint16_t value) { m_unsignedShort = value; }

    int testLong() { return m_long; }
    void setTestLong(int value) { m_long = value; }
    int testEnforceRangeLong() { return m_long; }
    void setTestEnforceRangeLong(int value) { m_long = value; }
    int testClampLong() { return m_long; }
    void setTestClampLong(int value) { m_long = value; }
    unsigned testUnsignedLong() { return m_unsignedLong; }
    void setTestUnsignedLong(unsigned value) { m_unsignedLong = value; }
    unsigned testEnforceRangeUnsignedLong() { return m_unsignedLong; }
    void setTestEnforceRangeUnsignedLong(unsigned value) { m_unsignedLong = value; }
    unsigned testClampUnsignedLong() { return m_unsignedLong; }
    void setTestClampUnsignedLong(unsigned value) { m_unsignedLong = value; }

    long long testLongLong() { return m_longLong; }
    void setTestLongLong(long long value) { m_longLong = value; }
    long long testEnforceRangeLongLong() { return m_longLong; }
    void setTestEnforceRangeLongLong(long long value) { m_longLong = value; }
    long long testClampLongLong() { return m_longLong; }
    void setTestClampLongLong(long long value) { m_longLong = value; }
    unsigned long long testUnsignedLongLong() { return m_unsignedLongLong; }
    void setTestUnsignedLongLong(unsigned long long value) { m_unsignedLongLong = value; }
    unsigned long long testEnforceRangeUnsignedLongLong() { return m_unsignedLongLong; }
    void setTestEnforceRangeUnsignedLongLong(unsigned long long value) { m_unsignedLongLong = value; }
    unsigned long long testClampUnsignedLongLong() { return m_unsignedLongLong; }
    void setTestClampUnsignedLongLong(unsigned long long value) { m_unsignedLongLong = value; }

    float testFloat() const { return m_float; }
    void setTestFloat(float testFloat) { m_float = testFloat; }
    float testUnrestrictedFloat() const { return m_unrestrictedFloat; }
    void setTestUnrestrictedFloat(float unrestrictedFloat) { m_unrestrictedFloat = unrestrictedFloat; }

    const String& testString() const { return m_string; }
    void setTestString(const String& string) { m_string = string; }
    const String& testUSVString() const { return m_usvstring; }
    void setTestUSVString(const String& usvstring) { m_usvstring = usvstring; }
    const String& testByteString() const { return m_byteString; }
    void setTestByteString(const String& byteString) { m_byteString = byteString; }
    const String& testTreatNullAsEmptyString() const { return m_treatNullAsEmptyString; }
    void setTestTreatNullAsEmptyString(const String& string) { m_treatNullAsEmptyString = string; }

    const Vector<KeyValuePair<String, int>>& testLongRecord() const { return m_longRecord; }
    void setTestLongRecord(const Vector<KeyValuePair<String, int>>& value) { m_longRecord = value; }
    const Vector<KeyValuePair<String, Ref<Node>>>& testNodeRecord() const { return m_nodeRecord; }
    void setTestNodeRecord(const Vector<KeyValuePair<String, Ref<Node>>>& value) { m_nodeRecord = value; }
    const Vector<KeyValuePair<String, Vector<String>>>& testSequenceRecord() const { return m_sequenceRecord; }
    void setTestSequenceRecord(const Vector<KeyValuePair<String, Vector<String>>>& value) { m_sequenceRecord = value; }

    using TestUnion = std::variant<String, int, bool, RefPtr<Node>, Vector<int>>;
    const TestUnion& testUnion() const { return m_union; }
    void setTestUnion(TestUnion&& value) { m_union = value; }

    const Dictionary& testDictionary() const { return m_testDictionary; }
    void setTestDictionary(Dictionary&& dictionary) { m_testDictionary = dictionary; }
    

    using TestClampUnion = std::variant<String, int, Vector<int>>;
    const TestClampUnion& testClampUnion() const { return m_clampUnion; }
    void setTestClampUnion(const TestClampUnion& value) { m_clampUnion = value; }

    using TestEnforceRangeUnion = std::variant<String, int, Vector<int>>;
    const TestEnforceRangeUnion& testEnforceRangeUnion() const { return m_enforceRangeUnion; }
    void setTestEnforceRangeUnion(const TestEnforceRangeUnion& value) { m_enforceRangeUnion = value; }

    using TestTreatNullAsEmptyStringUnion = std::variant<String, int, Vector<String>>;
    const TestTreatNullAsEmptyStringUnion& testTreatNullAsEmptyStringUnion() const { return m_treatNullAsEmptyStringUnion; }
    void setTestTreatNullAsEmptyStringUnion(const TestTreatNullAsEmptyStringUnion& value) { m_treatNullAsEmptyStringUnion = value; }

    double testImpureNaNUnrestrictedDouble() const { return std::bit_cast<double>(0xffff000000000000ll); }
    double testImpureNaN2UnrestrictedDouble() const { return std::bit_cast<double>(0x7ff8000000000001ll); }
    double testQuietNaNUnrestrictedDouble() const { return std::numeric_limits<double>::quiet_NaN(); }
    double testPureNaNUnrestrictedDouble() const { return JSC::PNaN; }

private:
    TypeConversions() = default;

    int8_t m_byte { 0 };
    uint8_t m_octet { 0 };
    int16_t m_short { 0 };
    uint16_t m_unsignedShort { 0 };
    int m_long { 0 };
    unsigned m_unsignedLong { 0 };
    long long m_longLong { 0 };
    unsigned long long m_unsignedLongLong { 0 };
    float m_float { 0 };
    float m_unrestrictedFloat { 0 };
    String m_string;
    String m_usvstring;
    String m_byteString;
    String m_treatNullAsEmptyString;
    Vector<KeyValuePair<String, int>> m_longRecord;
    Vector<KeyValuePair<String, Ref<Node>>> m_nodeRecord;
    Vector<KeyValuePair<String, Vector<String>>> m_sequenceRecord;

    Dictionary m_testDictionary;

    TestUnion m_union;
    TestClampUnion m_clampUnion;
    TestEnforceRangeUnion m_enforceRangeUnion;
    TestTreatNullAsEmptyStringUnion m_treatNullAsEmptyStringUnion;
};

} // namespace WebCore
