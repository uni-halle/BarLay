package cmd

import "github.com/spf13/cobra"

var layoutCmd = &cobra.Command{
	Use:   "layout",
	Short: "commands related to layouts of barcodes",
}

func init() {
	rootCmd.AddCommand(layoutCmd)
}
