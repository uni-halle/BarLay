package cmd

import "testing"

func TestHammingDistance(t *testing.T) {
	testCases := []struct {
		a        string
		b        string
		expected int
	}{
		{
			a:        "101010",
			b:        "010101",
			expected: 6,
		},
	}

	for _, testCase := range testCases {
		actual := HammingDistance(testCase.a, testCase.b)
		if actual != testCase.expected {
			t.Errorf("%s and %s should have distance %d but had %d", testCase.a, testCase.b, testCase.expected, actual)
		}
	}
}
