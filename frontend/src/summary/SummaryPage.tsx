import React from 'react';
import { Header } from './Header';
import { InfoComp } from './InfoComp';
import { LeftNavigation } from './LeftNavigation';

function SummaryPage() {
  return (
    <div>
      <Header></Header>
      <LeftNavigation></LeftNavigation>
      <InfoComp></InfoComp>
    </div>
  );
}

export default SummaryPage;
