package cmd

import (
	"bufio"
	"fmt"
	"os"

	"github.com/spf13/cobra"
)

var layoutVerifyCmd = &cobra.Command{
	Use:   "verify",
	Short: "parses a stream of newline separated strings from stdin and verifies, that they are valid layouts for a given number of columns and rows",
	Run: func(cmd *cobra.Command, args []string) {
		columns, err := cmd.Flags().GetInt("columns")
		cobra.CheckErr(err)

		rows, err := cmd.Flags().GetInt("rows")
		cobra.CheckErr(err)

		foundError := false
		expectedLineCount := columns * rows

		different := make(map[string]struct{}, expectedLineCount)
		length := -1

		scanner := bufio.NewScanner(os.Stdin)
		lineCount := 0

		var (
			previousRow  []string
			currentRow   = make([]string, 0, columns)
			hammingSumN8 = 0
		)

		for scanner.Scan() {
			line := scanner.Text()

			if length == -1 {
				length = len(line)
			} else if len(line) != length {
				fmt.Printf("not all entries have the same size (the first one had %d but the entry no %d had length %d)\n", length, lineCount, len(line))
				foundError = true
			}

			_, alreadyExists := different[line]
			if alreadyExists {
				fmt.Printf("duplicate line %v(%s)\n", []byte(line), line)
				foundError = true
			}

			lineCount++

			currentRow = append(currentRow, line)
			x := len(currentRow) - 1

			if len(previousRow) > 0 {
				hammingSumN8 += HammingDistance(previousRow[x], line)

				if x > 0 {
					hammingSumN8 += HammingDistance(previousRow[x-1], line)
				}

				if x < columns-1 {
					hammingSumN8 += HammingDistance(previousRow[x+1], line)
				}
			}

			if x > 0 {
				hammingSumN8 += HammingDistance(currentRow[x-1], line)
			}

			if len(currentRow) >= columns {
				previousRow = currentRow
				currentRow = make([]string, 0, columns)
			}
		}

		if lineCount > expectedLineCount {
			fmt.Printf("too many lines (expected %d but got %d)\n", expectedLineCount, lineCount)
			foundError = true
		}

		if lineCount < expectedLineCount {
			fmt.Printf("not enough lines (expected %d but got %d)\n", expectedLineCount, lineCount)
			foundError = true
		}

		if foundError {
			os.Exit(1)
		} else {
			fmt.Printf(`
the input was a valid %dx%d layout with unique entries of length %d
the sum of hamming distances in an 8-neighborhood was %d (unidirectional) and %d (bidirectional)
`, rows, columns, length, hammingSumN8, 2*hammingSumN8)
		}
	},
}

func init() {
	layoutCmd.AddCommand(layoutVerifyCmd)

	layoutVerifyCmd.Flags().IntP("columns", "c", 1024, "columns of the layout")
	layoutVerifyCmd.MarkFlagRequired("columns")

	layoutVerifyCmd.Flags().IntP("rows", "r", 1024, "rows of the layout")
	layoutVerifyCmd.MarkFlagRequired("rows")
}

func HammingDistance(a, b string) (distance int) {
	if len(a) != len(b) {
		panic(fmt.Errorf("a and b have different lengths ('%s', '%s')", a, b))
	}

	for i, aChar := range []byte(a) {
		bChar := b[i]

		if aChar != bChar {
			distance++
		}
	}

	return distance
}
